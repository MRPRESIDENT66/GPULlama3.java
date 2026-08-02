package org.beehive.gpullama3.inference;

import org.beehive.gpullama3.auxiliary.Parallel;
import org.beehive.gpullama3.inference.state.Qwen2MoEState;
import org.beehive.gpullama3.inference.weights.standard.*;
import org.beehive.gpullama3.model.qwen2.Qwen2MoE;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tensor.standard.FloatTensor;
import org.beehive.gpullama3.inference.state.Phi3State;
import org.beehive.gpullama3.inference.state.State;
import org.beehive.gpullama3.inference.weights.tornado.TornadoWeights;
import org.beehive.gpullama3.model.Configuration;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.model.granite.GraniteConfiguration;
import org.beehive.gpullama3.model.devstral.DevstralConfiguration;
import org.beehive.gpullama3.model.phi3.Phi3Configuration;
import org.beehive.gpullama3.model.qwen2.Qwen2Configuration;
import org.beehive.gpullama3.model.qwen3.Qwen3Configuration;
import org.beehive.gpullama3.tornadovm.TornadoVMMasterPlan;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import org.beehive.gpullama3.validation.MoECorrectnessTrace;

import java.lang.foreign.MemorySegment;

/**
 * Low-level operations for model inference.
 *
 * <p>
 * This class provides core computational operations such as RMS normalization and forward passes through model layers. It supports both CPU and GPU implementations.
 * </p>
 *
 */

public final class InferenceCore {

    private InferenceCore() {
        // prevent instantiation
    }

    public static void rmsnorm(FloatTensor out, FloatTensor x, FloatTensor weight, int offset, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(offset, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(offset, size, (value, index) -> weight.getFloat(index % size) * (finalss * x.getFloat(index)));
    }

    public static FloatTensor forwardJava(Model model, State state, int token, int position) {
        // a few convenience variables
        final Configuration config = model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position

            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim;
            // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection
            state.x.addInPlace(state.xb);
        }

        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    /**
     * Forward pass for Devstral 2 models where head_dim != dim/num_heads.
     * Q projection outputs qDim (num_heads * head_dim) instead of dim.
     */
    public static FloatTensor forwardJavaDevstral(Model model, State state, int token, int position) {
        final DevstralConfiguration config = (DevstralConfiguration) model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();       // 128 (independent head_dim)
        int qDim = config.qDim();               // 4096 = 32 * 128
        int kvDim = config.kvDim();              // 1024 = 8 * 128
        int kvMul = config.kvMul();
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        for (int l = 0; l < config.numberOfLayers(); l++) {
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            weights.wq[l].matmul(state.xb, state.q, qDim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE over qDim (not dim)
            for (int i = 0; i < qDim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k;
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength();

                for (int t = 0; t <= position; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset, position + 1);

                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // O projection: input qDim, output dim
            weights.wo[l].matmul(state.xb, state.xb2, dim, qDim);

            state.x.addInPlace(state.xb2);

            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            state.hb.multiplyInPlace(state.hb2);

            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());
            state.x.addInPlace(state.xb);
        }

        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    public static FloatTensor forwardJavaQwen2MoE(Model model, State state, int token, int position) {
        final Qwen2MoEConfiguration config = (Qwen2MoEConfiguration) model.configuration();
        final Qwen2MoEStandardWeights weights = (Qwen2MoEStandardWeights) model.weights();
        final Qwen2MoEState moeState = (Qwen2MoEState) state;
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            final int curLayer = l;
            rmsnorm(state.xb, state.x, weights.rms_att_weight[curLayer], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // qkv additions with qkv bias
            state.q.addInPlace(weights.q_bias[curLayer]);
            state.k.addInPlace(weights.k_bias[curLayer]);
            state.v.addInPlace(weights.v_bias[curLayer]);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            for (int h = 0; h < config.numberOfHeads(); ++h) {
                int rotn = h < config.numberOfKeyValueHeads() ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                int poffset = h * headSize;
                for (int i0 = 0; i0 < headSize; i0 += 2) {
                    int ic = i0 / 2;
                    float fcr = weights.freq_cis_real.getFloat((position) * (headSize / 2) + ic);
                    float fci = weights.freq_cis_imag.getFloat((position) * (headSize / 2) + ic);
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = (vi == 0) ? state.q : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + headSize / 2);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + headSize / 2, v0 * fci + v1 * fcr);
                    }
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[curLayer], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[curLayer], position * kvDim, kvDim);

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;C
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // MoE FFN pre-normalization
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[curLayer], 0, dim, config.rmsNormEps());

            int numberOfExperts = config.numberOfExperts();
            int topK = config.numberOfExpertsUsed();
            int expertHiddenDim = config.moeHiddenDim();

            // Compute routing probabilities over all experts, then select top-k.
            // Qwen1.5-MoE uses norm_topk_prob=false: each selected expert's routing weight is
            // its probability over all experts without rescaling the top-k weights to sum to one.
            weights.routerGate[l].matmul(state.xb, moeState.routerLogits, numberOfExperts, dim);
            float[] rawRouterLogits = null;
            if (MoECorrectnessTrace.isEnabled()) {
                rawRouterLogits = new float[numberOfExperts];
                for (int expert = 0; expert < numberOfExperts; expert++) {
                    rawRouterLogits[expert] = moeState.routerLogits.getFloat(expert);
                }
            }
            moeState.routerLogits.softmaxInPlace(0, numberOfExperts);

            int[] selectedExperts = new int[topK];
            float[] routingWeights = new float[topK];
            for (int i = 0; i < topK; i++) {
                float best = Float.NEGATIVE_INFINITY;
                int index = -1;
                for (int j = 0; j < numberOfExperts; j++) {
                    if (moeState.routerLogits.getFloat(j) > best) {
                        best = moeState.routerLogits.getFloat(j);
                        index = j;
                    }
                }
                selectedExperts[i] = index;
                routingWeights[i] = best;
                moeState.routerLogits.setFloat(index, Float.NEGATIVE_INFINITY);
            }
            if (MoECorrectnessTrace.isEnabled()) {
                MoECorrectnessTrace.recordCpuRouter(position, l, rawRouterLogits,
                        selectedExperts, routingWeights);
            }

            // Compute each selected expert and accumulate its weighted output.
            for (int j = 0; j < topK; j++) {
                int expert = selectedExperts[j];
                int gateUpOffset = expert * expertHiddenDim * dim;
                int downOffset = expert * dim * expertHiddenDim;
                matmulExpert(weights.gateExps[l], gateUpOffset, state.xb, moeState.hbE, expertHiddenDim, dim);
                matmulExpert(weights.upExps[l], gateUpOffset, state.xb, moeState.hbE2, expertHiddenDim, dim);
                moeState.hbE.mapInPlace(v -> v / (float) (1.0 + Math.exp(-v)));
                moeState.hbE.multiplyInPlace(moeState.hbE2);
                matmulExpert(weights.downExps[l], downOffset, moeState.hbE, moeState.yTmp, dim, expertHiddenDim);
                state.x.saxpyInPlace(0, moeState.yTmp, 0, dim, routingWeights[j]);
            }

            // Compute the always-on shared expert.
            int sharedExpertHiddenDim = config.sharedExpertHiddenDim();
            weights.sharedGate[l].matmul(state.xb, moeState.hbS, sharedExpertHiddenDim, dim);
            weights.sharedUp[l].matmul(state.xb, moeState.hbS2, sharedExpertHiddenDim, dim);
            moeState.hbS.mapInPlace(v -> v / (float) (1.0 + Math.exp(-v)));
            moeState.hbS.multiplyInPlace(moeState.hbS2);
            weights.sharedDown[l].matmul(moeState.hbS, moeState.yTmp, dim, sharedExpertHiddenDim);

            // Gate the shared expert output.
            float gateScore = weights.sharedGateInp[l].dot(0, state.xb, 0, dim);
            float sharedExpertWeight = 1f / (1f + (float) Math.exp(-gateScore));
            state.x.saxpyInPlace(0, moeState.yTmp, 0, dim, sharedExpertWeight);
        }

        // final rmsnorm + classifier (same as dense Qwen2)
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);
        MoECorrectnessTrace.recordCpuLogits(position, state.logits);

        return state.logits;
    }

    /**
     * Like {@link FloatTensor#matmul}, but reads the weight matrix starting at element
     * offset {@code base} instead of 0 — so it can target ONE expert inside a stacked
     * {@code [nExpert × d0 × d1]} tensor. out[d0] = (d0×d1 sub-matrix at base) · in[d1].
     */
    private static void matmulExpert(FloatTensor w, int base, FloatTensor in, FloatTensor out, int d0, int d1) {
        Parallel.parallelFor(0, d0, i -> out.setFloat(i, w.dot(base + i * d1, in, 0, d1)));
    }

    public static FloatTensor forwardJavaQwen2(Model model, State state, int token, int position) {
        final Qwen2Configuration config = (Qwen2Configuration) model.configuration();
        final Qwen2StandardWeights weights = (Qwen2StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            final int curLayer = l;
            rmsnorm(state.xb, state.x, weights.rms_att_weight[curLayer], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // qkv additions with qkv bias
            state.q.addInPlace(weights.q_bias[curLayer]);
            state.k.addInPlace(weights.k_bias[curLayer]);
            state.v.addInPlace(weights.v_bias[curLayer]);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            for (int h = 0; h < config.numberOfHeads(); ++h) {
                int rotn = h < config.numberOfKeyValueHeads() ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                int poffset = h * headSize;
                for (int i0 = 0; i0 < headSize; i0 += 2) {
                    int ic = i0 / 2;
                    float fcr = weights.freq_cis_real.getFloat((position) * (headSize / 2) + ic);
                    float fci = weights.freq_cis_imag.getFloat((position) * (headSize / 2) + ic);
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = (vi == 0) ? state.q : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + headSize / 2);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + headSize / 2, v0 * fci + v1 * fcr);
                    }
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[curLayer], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[curLayer], position * kvDim, kvDim);

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;C
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[curLayer], 0, dim, config.rmsNormEps());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection
            state.x.addInPlace(state.xb);

        }

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    public static FloatTensor forwardJavaQwen3(Model model, State state, int token, int position) {
        // a few convenience variables
        final Qwen3Configuration config = (Qwen3Configuration) model.configuration();
        final Qwen3StandardWeights weights = (Qwen3StandardWeights) model.weights();
        int dim = config.dim();
        int nHeadKv = config.numberOfKeyValueHeads(); // n_head_kv = numberOfKeyValueHeads
        int nEmbdHeadK = config.numberOfHeadsKey(); // n_embd_head_k = n_embd / n_head; %s.attention.key_length
        int nEmbdHeadV = config.numberOfHeadsValue(); // n_embd_head_v = n_embd / n_head; %s.attention.value_length
        int nEmbdVGqa = nEmbdHeadV * nHeadKv; // n_embd_v_gqa = n_embd_head_v * n_head_kv
        int nEmbdHead = nEmbdHeadV;
        int nEmbdGqa = nEmbdVGqa;
        int gqa = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(nEmbdHead);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            final int curLayer = l;
            rmsnorm(state.xb, state.x, weights.rms_att_weight[curLayer], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[curLayer].matmul(state.xb, state.q, nEmbdHeadK * config.numberOfHeads(), dim);
            weights.wk[curLayer].matmul(state.xb, state.k, nEmbdGqa, dim);
            weights.wv[curLayer].matmul(state.xb, state.v, nEmbdGqa, dim);

            // Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            for (int i = 0; i < config.numberOfHeads(); i++) {
                rmsnorm(state.q, state.q, weights.attnQNorm[curLayer], i * nEmbdHead, nEmbdHead, config.rmsNormEps());
            }
            // Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            for (int i = 0; i < config.numberOfKeyValueHeads(); i++) {
                rmsnorm(state.k, state.k, weights.attnKNorm[curLayer], i * nEmbdHead, nEmbdHead, config.rmsNormEps());
            }

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            for (int h = 0; h < config.numberOfHeads(); ++h) {
                int rotn = h < config.numberOfKeyValueHeads() ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                int poffset = h * nEmbdHead;
                int nComplEmbdHead = nEmbdHead / 2;
                for (int ic = 0; ic < nComplEmbdHead; ic++) {
                    float fcr = weights.freq_cis_real.getFloat(position * nComplEmbdHead + ic);
                    float fci = weights.freq_cis_imag.getFloat(position * nComplEmbdHead + ic);
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = (vi == 0) ? state.q : state.k; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(poffset + ic);
                        float v1 = vec.getFloat(poffset + ic + nComplEmbdHead);
                        vec.setFloat(poffset + ic, v0 * fcr - v1 * fci);
                        vec.setFloat(poffset + ic + nComplEmbdHead, v0 * fci + v1 * fcr);
                    }
                }
            }

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim;
            // kv cache layer offset for convenience
            state.k.copyTo(0, state.keyCache[curLayer], position * nEmbdGqa, nEmbdGqa);
            state.v.copyTo(0, state.valueCache[curLayer], position * nEmbdGqa, nEmbdGqa);

            // multihead attention. iterate over all heads
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                // get the query vector for this head
                int qOffset = h * nEmbdHead;
                // attention scores for this head
                int attOffset = h * config.contextLength();

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position; t++) {
                    // get the key vector for this head and at this timestep
                    int keyCacheOffset = /* loff + */ (t * nEmbdGqa + (h / gqa) * nEmbdHead);
                    // calculate the attention score as the dot product of q and k
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, nEmbdHeadK);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att.setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att.softmaxInPlace(attOffset, position + 1); // position + 0 + 1

                // weighted sum of the values, store back into xb
                int xbOffset = h * nEmbdHeadV;
                state.xb.fillInPlace(xbOffset, nEmbdHeadV, 0f);

                for (int t = 0; t <= position; t++) {
                    // get the value vector for this head and at this timestep
                    int vOffset = /* loff + */ t * nEmbdGqa + (h / gqa) * nEmbdHeadV;
                    // get the attention weight for this timestep
                    float a = state.att.getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, nEmbdHeadV, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, nEmbdHeadK * config.numberOfHeads());

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[curLayer], 0, dim, config.rmsNormEps());

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            // elementwise multiply with w3(x)
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection
            state.x.addInPlace(state.xb);
        }

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    public static FloatTensor forwardJavaPhi3(Model model, Phi3State state, int token, int position) {
        Phi3Configuration config = (Phi3Configuration) model.configuration();
        Phi3StandardWeights weights = (Phi3StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads(); // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);

        // Phi3: op_size = num_heads * head_dim + 2 * (num_key_value_heads * head_dim)
        final int opSize = dim + 2 * (config.numberOfKeyValueHeads() * headSize);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            weights.wqkv[l].matmul(state.xb, state.qkv, opSize, dim);
            state.qkv.copyTo(0, state.q, 0, dim);
            // key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
            state.qkv.copyTo(dim, state.k, 0, config.numberOfKeyValueHeads() * headSize);
            // value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
            state.qkv.copyTo(dim + config.numberOfKeyValueHeads() * headSize, state.v, 0, config.numberOfKeyValueHeads() * headSize);

            int dimHalf = headSize / 2;
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                int base = i - head_dim;
                int ic = base + head_dim / 2;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k; // the vector to rotate (query or key)
                    float v0 = vec.getFloat(ic);
                    float v1 = vec.getFloat(ic + dimHalf);
                    vec.setFloat(ic, v0 * fcr - v1 * fci);
                    vec.setFloat(ic + dimHalf, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step (position) to our kv cache
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                int qOffset = h * headSize;

                int attOffset = h * config.contextLength();

                for (int t = 0; t <= position; t++) {
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset, position + 1);

                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection back into x
            state.x.addInPlace(state.xb2);

            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            weights.wGateUp[l].matmul(state.xb, state.hb, 2 * config.hiddenDim(), dim);
            copyChunk(state.hb, state.hbG, 2 * config.hiddenDim(), config.hiddenDim(), 2, 0);
            copyChunk(state.hb, state.hbU, 2 * config.hiddenDim(), config.hiddenDim(), 2, 1);

            state.hbG.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));

            state.hbU.multiplyInPlace(state.hbG);

            weights.wDown[l].matmul(state.hbU, state.xb, dim, config.hiddenDim());

            state.x.addInPlace(state.xb);
        }

        // final rmsnorm
        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        // classifier into logits
        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        return state.logits;
    }

    /**
     * Forward pass for Granite models with µP scaling factors applied.
     * <p>
     * Granite uses the same transformer architecture as Llama but with maximal update parameterization (µP)
     * scaling factors applied at specific points:
     * <ul>
     * <li>Embedding scaling: multiply embeddings after lookup</li>
     * <li>Attention scaling: use custom multiplier instead of 1/sqrt(headDim)</li>
     * <li>Residual scaling: multiply residual connections</li>
     * <li>Logit scaling: divide logits by the scaling factor</li>
     * </ul>
     */
    public static FloatTensor forwardGranite(Model model, State state, int token, int position) {
        final GraniteConfiguration config = (GraniteConfiguration) model.configuration();
        final StandardWeights weights = (StandardWeights) model.weights();
        int dim = config.dim();
        int headSize = config.headSize();
        int kvDim = (config.dim() * config.numberOfKeyValueHeads()) / config.numberOfHeads();
        int kvMul = config.numberOfHeads() / config.numberOfKeyValueHeads();
        float attentionScale = config.attentionScale();
        float residualScale = config.residualScale();
        float embeddingScale = config.embeddingScale();
        float logitScale = config.logitScale();

        // copy the token embedding into x
        weights.token_embedding_table.copyTo(token * dim, state.x, 0, dim);
        // Apply Granite embedding scaling
        state.x.mapInPlace(v -> v * embeddingScale);

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers(); l++) {
            // attention rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_att_weight[l], 0, dim, config.rmsNormEps());

            // qkv matmuls for this position
            weights.wq[l].matmul(state.xb, state.q, dim, dim);
            weights.wk[l].matmul(state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % headSize;
                float fcr = weights.freq_cis_real.getFloat(position * (headSize / 2) + (head_dim / 2));
                float fci = weights.freq_cis_imag.getFloat(position * (headSize / 2) + (head_dim / 2));
                int rotn = i < kvDim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    FloatTensor vec = v == 0 ? state.q : state.k;
                    float v0 = vec.getFloat(i);
                    float v1 = vec.getFloat(i + 1);
                    vec.setFloat(i, v0 * fcr - v1 * fci);
                    vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                }
            }

            // save key,value at this time step to kv cache
            state.k.copyTo(0, state.keyCache[l], position * kvDim, kvDim);
            state.v.copyTo(0, state.valueCache[l], position * kvDim, kvDim);

            int curLayer = l;

            // multihead attention with Granite attention scaling
            Parallel.parallelFor(0, config.numberOfHeads(), h -> {
                int qOffset = h * headSize;
                int attOffset = h * config.contextLength();

                for (int t = 0; t <= position; t++) {
                    int keyCacheOffset = t * kvDim + (h / kvMul) * headSize;
                    float score = state.q.dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    // Granite uses custom attention multiplier instead of 1/sqrt(headSize)
                    score *= attentionScale;
                    state.att.setFloat(attOffset + t, score);
                }

                state.att.softmaxInPlace(attOffset, position + 1);

                int xbOffset = h * headSize;
                state.xb.fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position; t++) {
                    int vOffset = t * kvDim + (h / kvMul) * headSize;
                    float a = state.att.getFloat(attOffset + t);
                    state.xb.saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(state.xb, state.xb2, dim, dim);

            // residual connection with Granite scaling
            state.xb2.mapInPlace(v -> v * residualScale);
            state.x.addInPlace(state.xb2);

            // ffn rmsnorm
            rmsnorm(state.xb, state.x, weights.rms_ffn_weight[l], 0, dim, config.rmsNormEps());

            // FFN: self.w2(F.silu(self.w1(x)) * self.w3(x))
            weights.w1[l].matmul(state.xb, state.hb, config.hiddenDim(), dim);
            weights.w3[l].matmul(state.xb, state.hb2, config.hiddenDim(), dim);

            // SwiGLU non-linearity
            state.hb.mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            state.hb.multiplyInPlace(state.hb2);

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(state.hb, state.xb, dim, config.hiddenDim());

            // residual connection with Granite scaling
            state.xb.mapInPlace(v -> v * residualScale);
            state.x.addInPlace(state.xb);
        }

        rmsnorm(state.x, state.x, weights.rms_final_weight, 0, dim, config.rmsNormEps());

        weights.wcls.matmul(state.x, state.logits, config.vocabularySize(), dim);

        // Apply Granite logit scaling (divide by the scaling factor)
        state.logits.mapInPlace(v -> v * logitScale);

        return state.logits;
    }

    static void copyChunk(FloatTensor in, FloatTensor out, int dim1In, int dim1Out, int nChunks, int chunkNo) {
        assert (dim1In == dim1Out * nChunks);
        final int startOffsetInDim1 = chunkNo * dim1Out;
        Parallel.parallelFor(0, dim1Out, i -> {
            out.setFloat(i, in.getFloat(startOffsetInDim1 + i));
        });
    }

    /**
     * Performs the initial embedding lookup and triggers the TornadoVM accelerated forward pass for an LLM token.
     *
     * <p>This method handles the first phase of processing a token through the transformer model:
     * <ol>
     * <li>Copies the token embedding from the model's embedding table to the state's buffer</li>
     * <li>Delegates the transformer layer processing to TornadoVM through the master plan</li>
     * </ol>
     *
     * <p>The token embedding lookup happens on the CPU using {@link MemorySegment} operations,
     * while the subsequent transformer layers processing is offloaded to the accelerator through
     * TornadoVM for improved performance.
     *
     * @param model
     *     The Llama model containing weights and configuration parameters
     * @param state
     *     The current execution state holding input/output tensors and temporary buffers
     * @param token
     *     The input token ID to process
     * @param position
     *     The position of this token in the sequence context window
     * @param tornadoVMMasterPlan
     *     The execution plan for TornadoVM acceleration
     * @return FloatTensor containing the output logits for token prediction
     */
    public static FloatArray forwardTornadoVM(Model model, State state, int token, int position, TornadoVMMasterPlan tornadoVMMasterPlan) {
        final Configuration configuration = model.configuration();
        final TornadoWeights weights = (TornadoWeights) model.weights();

        switch (weights.getWeightType()) {
            case F16 -> {
                MemorySegment tokenEmbeddings = weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                int bytes = Short.BYTES;
                MemorySegment.copy(tokenEmbeddings, (long) token * configuration.dim() * bytes, state.embeddingX.getSegment(), 0, (long) configuration.dim() * bytes);
            }
            case Q8_0 -> {
                MemorySegment tokenEmbeddings = weights.getTokenEmbeddingTable().asByteArray().getSegment();
                int blockSize = 32;
                int Q8_0_BLOCK_BYTES = 34; // 2 bytes scale + 32 bytes quants
                int blocksPerToken = (configuration.dim() + blockSize - 1) / blockSize; // Ceiling division
                long bytesPerToken = (long) blocksPerToken * Q8_0_BLOCK_BYTES;

                MemorySegment.copy(tokenEmbeddings, (long) token * bytesPerToken, state.embeddingX.getSegment(), 0, bytesPerToken);

            }
            default -> throw new IllegalArgumentException("Unsupported weight type: " + weights.getWeightType());
        }

        return tornadoVMMasterPlan.tornadoVMForwardDecode(position);
    }

}
