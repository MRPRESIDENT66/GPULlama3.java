package org.beehive.gpullama3.model.loader;

import org.beehive.gpullama3.inference.operation.RoPE;
import org.beehive.gpullama3.inference.weights.Weights;
import org.beehive.gpullama3.inference.weights.standard.Qwen2MoEStandardWeights;
import org.beehive.gpullama3.model.format.ChatFormat;
import org.beehive.gpullama3.model.qwen2.Qwen2MoE;
import org.beehive.gpullama3.model.qwen2.Qwen2MoEConfiguration;
import org.beehive.gpullama3.tensor.GGMLTensorEntry;
import org.beehive.gpullama3.tensor.GGUF;
import org.beehive.gpullama3.auxiliary.Pair;
import org.beehive.gpullama3.tensor.standard.ArrayFloatTensor;
import org.beehive.gpullama3.tokenizer.Qwen3Tokenizer;
import org.beehive.gpullama3.tokenizer.Tokenizer;
import org.beehive.gpullama3.tokenizer.Vocabulary;

import static org.beehive.gpullama3.model.loader.ModelLoader.*;

import java.nio.channels.FileChannel;
import java.util.Map;

public class Qwen2MoEModelLoader  extends AbstractModelLoader<Qwen2MoE, Qwen2MoEConfiguration> {
    @Override
    protected Vocabulary loadVocabulary(Map<String, Object> metadata) {
        return Vocabulary.loadQwen3Vocabulary(metadata);
    }

    @Override
    protected Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        boolean isDeepSeekR1DistillQwen = "DeepSeek-R1-Distill-Qwen".equals(metadata.get("general.basename"));
        return new Qwen3Tokenizer(metadata, vocabulary, isDeepSeekR1DistillQwen);
    }

    @Override
    protected Pair<float[], float[]> precomputeRopeFrequencies(Qwen2MoEConfiguration config) {
        return RoPE.precomputeFreqsCis(config.contextLengthModel(), config.headSize(), config.ropeTheta(), false, 8, 1, 3, 8192);
    }

    @Override
    protected Qwen2MoE createModel(Qwen2MoEConfiguration config, Tokenizer tokenizer, Weights weights) {
        ChatFormat.ChatTokens chatTokens = new ChatFormat.ChatTokens("<|im_start|>", "<|im_end|>", "", "<|end_of_text|>", "<|endoftext|>");
        return new Qwen2MoE(config, tokenizer, weights, ChatFormat.create(tokenizer, chatTokens));
    }

    public Qwen2MoEModelLoader(FileChannel fileChannel, GGUF gguf, int contextLength, boolean useTornadovm) {
        super(fileChannel, gguf, contextLength, useTornadovm);
    }

    @Override
    protected Qwen2MoEConfiguration createConfiguration(Map<String, Object> metadata) {
        int modelContextLength = (int) metadata.get("qwen2moe.context_length");
        int finalContextLength = (contextLength < 0 || modelContextLength < contextLength) ? modelContextLength : contextLength;
        int numberOfKeyValueHeads = (int) metadata.get("qwen2moe.attention.head_count_kv");
        int vocabSize = vocabulary.size();

        int moeHiddenDim = gguf.getTensorInfos().get("blk.0.ffn_down_exps.weight").dimensions()[0];

        return new Qwen2MoEConfiguration(
                getModelQuantization(metadata),                          // quantization
                (int) metadata.get("qwen2moe.embedding_length"),          // dim
                0,                                                       // hiddenDim
        (int) metadata.get("qwen2moe.block_count"),               // numberOfLayers
                (int) metadata.get("qwen2moe.attention.head_count"),      // numberOfHeads
                numberOfKeyValueHeads,                                     // numberOfKeyValueHeads
                numberOfKeyValueHeads,                                     // numberOfHeadsKey
                numberOfKeyValueHeads,                                     // numberOfHeadsValue
                vocabSize,                                                 // vocabularySize
                modelContextLength,                                        // contextLengthModel
                finalContextLength,                                        // contextLength
                (int) metadata.get("qwen2moe.expert_count"),               // numberOfExperts
                (int) metadata.get("qwen2moe.expert_used_count"),          // numberOfExpertsUsed
                moeHiddenDim,                                               // moeHiddenDim
                (int) metadata.get("qwen2moe.feed_forward_length"),        // sharedExpertHiddenDim
                false,                                                      // sharedWeights
                (float) metadata.get("qwen2moe.attention.layer_norm_rms_epsilon"), // rmsNormEps
                (float) metadata.get("qwen2moe.rope.freq_base")            // ropeTheta
);
    }

    @Override
    protected Weights createStandardWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2MoEConfiguration config,
                                            Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                            GGMLTensorEntry outputWeight) {

        final int nl = config.numberOfLayers();

        return new Qwen2MoEStandardWeights(
                loadTensor(tokenEmbeddings),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_q.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_k.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_v.bias")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                null, // w1: qwen2moe has no dense ffn_gate, only routed/shared experts below
                null, // w2: qwen2moe has no dense ffn_down
                null, // w3: qwen2moe has no dense ffn_up
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_exps.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up_exps.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down_exps.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_shexp.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_up_shexp.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_down_shexp.weight")),
                loadArrayOfTensors(nl, i -> tensorEntries.get("blk." + i + ".ffn_gate_inp_shexp.weight")),
                loadTensor(tensorEntries.get("output_norm.weight")),
                new ArrayFloatTensor(ropeFreqs.first()),
                new ArrayFloatTensor(ropeFreqs.second()),
                loadTensor(outputWeight),
                outputWeight.ggmlType()
        );
    }

    @Override
    protected Weights createTornadoVMWeights(Map<String, GGMLTensorEntry> tensorEntries, Qwen2MoEConfiguration config,
                                              Pair<float[], float[]> ropeFreqs, GGMLTensorEntry tokenEmbeddings,
                                              GGMLTensorEntry outputWeight) {
        throw new UnsupportedOperationException("GPU MoE weights not yet supported");
    }
}