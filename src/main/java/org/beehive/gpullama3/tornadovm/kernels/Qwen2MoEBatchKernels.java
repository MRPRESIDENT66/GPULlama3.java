package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Correctness-first GPU kernels for expert-grouped Qwen2-MoE batch prefill. */
public final class Qwen2MoEBatchKernels {

    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;
    private static final int ACTIVE_EXPERT_TOKEN_TILE = 4;

    private Qwen2MoEBatchKernels() {
    }

    /** Copies the final active batch row into the single-token logits vector. */
    public static void copyLastActiveTokenActivation(
            KernelContext context,
            FloatArray batchActivation,
            FloatArray decodeActivation,
            IntArray activeBatchSizeHolder,
            int dim) {
        int column = context.globalIdx;
        if (column < dim) {
            int lastRowOffset = (activeBatchSizeHolder.get(0) - 1) * dim;
            decodeActivation.set(column, batchActivation.get(lastRowOffset + column));
        }
    }

    /** Copies raw batched router scores before the softmax/Top-K task overwrites them. */
    public static void copyBatchedRouterLogits(
            KernelContext context,
            FloatArray source,
            FloatArray destination,
            IntArray activeBatchSizeHolder,
            int numberOfExperts) {
        int index = context.globalIdx;
        if (index < activeBatchSizeHolder.get(0) * numberOfExperts) {
            destination.set(index, source.get(index));
        }
    }

    /** Adds Qwen2 Q/K/V projection biases independently for every token. */
    public static void batchedQKVBias(
            KernelContext context,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray qBias,
            FloatArray kBias,
            FloatArray vBias,
            IntArray activeBatchSizeHolder,
            int qDim,
            int kvDim) {

        int index = context.globalIdx;
        int valuesPerToken = qDim + 2 * kvDim;
        if (index >= activeBatchSizeHolder.get(0) * valuesPerToken) {
            return;
        }

        int token = index / valuesPerToken;
        int withinToken = index % valuesPerToken;
        if (withinToken < qDim) {
            int column = withinToken;
            int output = token * qDim + column;
            q.set(output, q.get(output) + qBias.get(column));
        } else if (withinToken < qDim + kvDim) {
            int column = withinToken - qDim;
            int output = token * kvDim + column;
            k.set(output, k.get(output) + kBias.get(column));
        } else {
            int column = withinToken - qDim - kvDim;
            int output = token * kvDim + column;
            v.set(output, v.get(output) + vBias.get(column));
        }
    }

    /**
     * Applies Qwen2 split-half RoPE and writes the batched K/V cache.
     * Each pair is {@code (ic, ic + headSize/2)} within one attention head.
     */
    public static void batchedQwen2RoPEWithKVCache(
            KernelContext context,
            IntArray batchStartPosHolder,
            FloatArray q,
            FloatArray k,
            FloatArray v,
            FloatArray keyCache,
            FloatArray valueCache,
            IntArray activeBatchSizeHolder,
            int numberOfKeyValueHeads,
            int headSize,
            int kvDim,
            int layerIndex,
            int contextLength,
            int dim,
            float ropeTheta) {

        int globalIndex = context.globalIdx;
        int pairsPerToken = dim / 2;
        int token = globalIndex / pairsPerToken;
        int pair = globalIndex % pairsPerToken;
        int halfHead = headSize / 2;
        int head = pair / halfHead;
        int ic = pair % halfHead;
        int position = batchStartPosHolder.get(0) + token;
        int activeBatchSize = activeBatchSizeHolder.get(0);

        // A fixed WorkerGrid can include rows beyond the final partial chunk.
        // Clear their cache slots so a later decode never attends to stale data
        // left by warmup or a previous larger chunk.
        if (token >= activeBatchSize) {
            if (position < contextLength && head < numberOfKeyValueHeads) {
                int cacheOffset = layerIndex * contextLength * kvDim
                        + position * kvDim + head * headSize;
                keyCache.set(cacheOffset + ic, 0.0f);
                keyCache.set(cacheOffset + ic + halfHead, 0.0f);
                valueCache.set(cacheOffset + ic, 0.0f);
                valueCache.set(cacheOffset + ic + halfHead, 0.0f);
            }
            return;
        }

        float frequency = 1.0f / TornadoMath.pow(
                ropeTheta, (float) (ic * 2) / (float) headSize);
        float angle = position * frequency;
        float cosine = TornadoMath.cos(angle);
        float sine = TornadoMath.sin(angle);

        int qHeadOffset = token * dim + head * headSize;
        float q0 = q.get(qHeadOffset + ic);
        float q1 = q.get(qHeadOffset + ic + halfHead);
        q.set(qHeadOffset + ic, q0 * cosine - q1 * sine);
        q.set(qHeadOffset + ic + halfHead, q0 * sine + q1 * cosine);

        if (head < numberOfKeyValueHeads) {
            int kvHeadOffset = token * kvDim + head * headSize;
            float k0 = k.get(kvHeadOffset + ic);
            float k1 = k.get(kvHeadOffset + ic + halfHead);
            float rotatedK0 = k0 * cosine - k1 * sine;
            float rotatedK1 = k0 * sine + k1 * cosine;
            k.set(kvHeadOffset + ic, rotatedK0);
            k.set(kvHeadOffset + ic + halfHead, rotatedK1);

            int cacheOffset = layerIndex * contextLength * kvDim
                    + position * kvDim + head * headSize;
            keyCache.set(cacheOffset + ic, rotatedK0);
            keyCache.set(cacheOffset + ic + halfHead, rotatedK1);
            valueCache.set(cacheOffset + ic, v.get(kvHeadOffset + ic));
            valueCache.set(cacheOffset + ic + halfHead,
                    v.get(kvHeadOffset + ic + halfHead));
        }
    }

    /** Computes the FP32 router projection for every token and expert. */
    public static void batchedRouterProjection(
            KernelContext context,
            FloatArray input,
            FloatArray routerLogits,
            FloatArray routerWeights,
            IntArray activeBatchSizeHolder,
            int dim,
            int numberOfExperts,
            int localWorkGroupSize) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int token = groupId / numberOfExperts;
        int expert = groupId % numberOfExperts;
        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        float partial = 0.0f;
        int inputOffset = token * dim;
        int weightOffset = expert * dim;
        for (int column = localId; column < dim; column += localWorkGroupSize) {
            partial += input.get(inputOffset + column) * routerWeights.get(weightOffset + column);
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partial;
        context.localBarrier();
        reduceLocal(context, localSums, localId, localWorkGroupSize);
        if (localId == 0) {
            routerLogits.set(token * numberOfExperts + expert, localSums[0]);
        }
    }

    /** Applies softmax and selects Top-K experts independently for every token. */
    public static void batchedSoftmaxAndTopK(
            KernelContext context,
            FloatArray routerLogits,
            IntArray selectedExperts,
            FloatArray routingWeights,
            IntArray activeBatchSizeHolder,
            int numberOfExperts,
            int topK) {

        int token = context.globalIdx;
        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int logitsOffset = token * numberOfExperts;
        int assignmentOffset = token * topK;
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int expert = 0; expert < numberOfExperts; expert++) {
            float logit = routerLogits.get(logitsOffset + expert);
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        float sumExp = 0.0f;
        for (int expert = 0; expert < numberOfExperts; expert++) {
            sumExp += TornadoMath.exp(routerLogits.get(logitsOffset + expert) - maxLogit);
        }
        for (int expert = 0; expert < numberOfExperts; expert++) {
            float probability = TornadoMath.exp(routerLogits.get(logitsOffset + expert) - maxLogit) / sumExp;
            routerLogits.set(logitsOffset + expert, probability);
        }

        for (int slot = 0; slot < topK; slot++) {
            float best = Float.NEGATIVE_INFINITY;
            int bestExpert = -1;
            for (int expert = 0; expert < numberOfExperts; expert++) {
                float probability = routerLogits.get(logitsOffset + expert);
                if (probability > best) {
                    best = probability;
                    bestExpert = expert;
                }
            }
            selectedExperts.set(assignmentOffset + slot, bestExpert);
            routingWeights.set(assignmentOffset + slot, best);
            routerLogits.set(logitsOffset + bestExpert, Float.NEGATIVE_INFINITY);
        }
    }

    /**
     * Groups the B*TopK token-expert assignments by expert on the GPU.
     * One thread is intentional here: E and B*TopK are small metadata arrays,
     * and this avoids atomics while establishing the correct data flow.
     */
    public static void groupAssignmentsByExpert(
            KernelContext context,
            IntArray selectedExperts,
            IntArray expertCounts,
            IntArray expertOffsets,
            IntArray groupedAssignmentIds,
            IntArray groupedPositionByAssignment,
            IntArray activeBatchSizeHolder,
            int numberOfExperts,
            int topK) {

        if (context.globalIdx != 0) {
            return;
        }

        int assignmentCount = activeBatchSizeHolder.get(0) * topK;
        int writePosition = 0;
        for (int expert = 0; expert < numberOfExperts; expert++) {
            expertOffsets.set(expert, writePosition);
            int count = 0;
            for (int assignment = 0; assignment < assignmentCount; assignment++) {
                if (selectedExperts.get(assignment) == expert) {
                    groupedAssignmentIds.set(writePosition, assignment);
                    groupedPositionByAssignment.set(assignment, writePosition);
                    writePosition++;
                    count++;
                }
            }
            expertCounts.set(expert, count);
        }
        expertOffsets.set(numberOfExperts, writePosition);
    }

    /** Computes grouped routed Gate/Up projections and SwiGLU activations. */
    public static void groupedGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray input,
            IntArray selectedExperts,
            IntArray groupedAssignmentIds,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray groupedHidden,
            IntArray activeBatchSizeHolder,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int topK,
            int localWorkGroupSize) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int groupedPosition = groupId / moeHiddenDim;
        int row = groupId % moeHiddenDim;
        int assignmentCount = activeBatchSizeHolder.get(0) * topK;
        if (groupedPosition >= assignmentCount) {
            return;
        }

        int assignment = groupedAssignmentIds.get(groupedPosition);
        int token = assignment / topK;
        int expert = selectedExperts.get(assignment);
        if (expert < 0 || expert >= numberOfExperts) {
            return;
        }

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = (expert * moeHiddenDim + row) * blocksPerRow;
        int inputOffset = token * dim;
        // Every 32 threads consume the same Q8_0 block scale. Load each scale
        // once into local memory before the dot product instead of loading it
        // again for every quant in that block.
        float[] gateBlockScales = context.allocateFloatLocalArray(blocksPerRow);
        float[] upBlockScales = context.allocateFloatLocalArray(blocksPerRow);
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * Q8_0_BLOCK_BYTES;
            gateBlockScales[block] = gateExperts.getHalfFloat(blockByteOffset).getFloat32();
            upBlockScales[block] = upExperts.getHalfFloat(blockByteOffset).getFloat32();
        }
        context.localBarrier();
        float gatePartial = 0.0f;
        float upPartial = 0.0f;
        for (int column = localId; column < dim; column += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
            int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;
            float inputValue = input.get(inputOffset + column);
            int block = column / Q8_0_BLOCK_SIZE;
            gatePartial += gateExperts.get(quantOffset) * gateBlockScales[block] * inputValue;
            upPartial += upExperts.get(quantOffset) * upBlockScales[block] * inputValue;
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = gatePartial;
        context.localBarrier();
        reduceLocal(context, localSums, localId, localWorkGroupSize);
        float gate = localSums[0];
        context.localBarrier();

        localSums[localId] = upPartial;
        context.localBarrier();
        reduceLocal(context, localSums, localId, localWorkGroupSize);
        if (localId == 0) {
            float siluGate = gate / (1.0f + TornadoMath.exp(-gate));
            groupedHidden.set(groupedPosition * moeHiddenDim + row, siluGate * localSums[0]);
        }
    }

    /** Down-projects every grouped assignment into a race-free temporary buffer. */
    public static void groupedDownQ8_0(
            KernelContext context,
            FloatArray groupedHidden,
            FloatArray groupedDown,
            IntArray selectedExperts,
            IntArray groupedAssignmentIds,
            FloatArray routingWeights,
            ByteArray downExperts,
            IntArray activeBatchSizeHolder,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int topK,
            int localWorkGroupSize) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int groupedPosition = groupId / dim;
        int row = groupId % dim;
        int assignmentCount = activeBatchSizeHolder.get(0) * topK;
        if (groupedPosition >= assignmentCount) {
            return;
        }

        int assignment = groupedAssignmentIds.get(groupedPosition);
        int expert = selectedExperts.get(assignment);
        if (expert < 0 || expert >= numberOfExperts) {
            return;
        }

        int blocksPerRow = (moeHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = (expert * dim + row) * blocksPerRow;
        int hiddenOffset = groupedPosition * moeHiddenDim;
        float[] downBlockScales = context.allocateFloatLocalArray(blocksPerRow);
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * Q8_0_BLOCK_BYTES;
            downBlockScales[block] = downExperts.getHalfFloat(blockByteOffset).getFloat32();
        }
        context.localBarrier();
        float partial = 0.0f;
        for (int column = localId; column < moeHiddenDim; column += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
            int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;
            float weight = downExperts.get(quantOffset)
                    * downBlockScales[column / Q8_0_BLOCK_SIZE];
            partial += weight * groupedHidden.get(hiddenOffset + column);
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partial;
        context.localBarrier();
        reduceLocal(context, localSums, localId, localWorkGroupSize);
        if (localId == 0) {
            groupedDown.set(groupedPosition * dim + row,
                    routingWeights.get(assignment) * localSums[0]);
        }
    }

    /**
     * Processes several assignments belonging to one active expert together.
     * The work-group is mapped to one active expert and one output row; the
     * expert's weight column is then reused across a small token tile.
     */
    public static void activeExpertTiledGateUpQ8_0(
            KernelContext context,
            FloatArray input,
            IntArray activeExpertIds,
            IntArray activeExpertCount,
            IntArray expertCounts,
            IntArray expertOffsets,
            IntArray groupedAssignmentIds,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray groupedHidden,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int topK,
            int localWorkGroupSize) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int expertSlot = groupId / moeHiddenDim;
        int row = groupId % moeHiddenDim;
        if (expertSlot >= activeExpertCount.get(0)) {
            return;
        }

        int expert = activeExpertIds.get(expertSlot);
        if (expert < 0 || expert >= numberOfExperts) {
            return;
        }
        int assignmentStart = expertOffsets.get(expert);
        int assignmentEnd = assignmentStart + expertCounts.get(expert);
        float[] gatePartials = context.allocateFloatLocalArray(
                ACTIVE_EXPERT_TOKEN_TILE * localWorkGroupSize);
        float[] upPartials = context.allocateFloatLocalArray(
                ACTIVE_EXPERT_TOKEN_TILE * localWorkGroupSize);
        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = (expert * moeHiddenDim + row) * blocksPerRow;

        for (int tileStart = assignmentStart; tileStart < assignmentEnd;
                tileStart += ACTIVE_EXPERT_TOKEN_TILE) {
            int tokenCount = assignmentEnd - tileStart;
            if (tokenCount > ACTIVE_EXPERT_TOKEN_TILE) {
                tokenCount = ACTIVE_EXPERT_TOKEN_TILE;
            }
            for (int tileToken = 0; tileToken < ACTIVE_EXPERT_TOKEN_TILE; tileToken++) {
                gatePartials[tileToken * localWorkGroupSize + localId] = 0.0f;
                upPartials[tileToken * localWorkGroupSize + localId] = 0.0f;
            }
            for (int column = localId; column < dim; column += localWorkGroupSize) {
                int blockByteOffset = (rowBlockOffset + column / Q8_0_BLOCK_SIZE)
                        * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;
                float gateWeight = gateExperts.get(quantOffset)
                        * gateExperts.getHalfFloat(blockByteOffset).getFloat32();
                float upWeight = upExperts.get(quantOffset)
                        * upExperts.getHalfFloat(blockByteOffset).getFloat32();
                for (int tileToken = 0; tileToken < tokenCount; tileToken++) {
                    int assignment = groupedAssignmentIds.get(tileStart + tileToken);
                    int token = assignment / topK;
                    float inputValue = input.get(token * dim + column);
                    gatePartials[tileToken * localWorkGroupSize + localId]
                            += gateWeight * inputValue;
                    upPartials[tileToken * localWorkGroupSize + localId]
                            += upWeight * inputValue;
                }
            }
            for (int tileToken = 0; tileToken < ACTIVE_EXPERT_TOKEN_TILE; tileToken++) {
                int base = tileToken * localWorkGroupSize;
                reduceLocalAt(context, gatePartials, base, localId, localWorkGroupSize);
                reduceLocalAt(context, upPartials, base, localId, localWorkGroupSize);
                if (localId == 0 && tileToken < tokenCount) {
                    float gate = gatePartials[base];
                    float up = upPartials[base];
                    float siluGate = gate / (1.0f + TornadoMath.exp(-gate));
                    int groupedPosition = tileStart + tileToken;
                    groupedHidden.set(groupedPosition * moeHiddenDim + row,
                            siluGate * up);
                }
            }
        }
    }

    /** Down-projects active-expert token tiles and writes routing-weighted output. */
    public static void activeExpertTiledDownQ8_0(
            KernelContext context,
            FloatArray groupedHidden,
            FloatArray groupedDown,
            IntArray activeExpertIds,
            IntArray activeExpertCount,
            IntArray expertCounts,
            IntArray expertOffsets,
            IntArray groupedAssignmentIds,
            FloatArray routingWeights,
            ByteArray downExperts,
            int dim,
            int moeHiddenDim,
            int numberOfExperts,
            int topK,
            int localWorkGroupSize) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int expertSlot = groupId / dim;
        int row = groupId % dim;
        if (expertSlot >= activeExpertCount.get(0)) {
            return;
        }

        int expert = activeExpertIds.get(expertSlot);
        if (expert < 0 || expert >= numberOfExperts) {
            return;
        }
        int assignmentStart = expertOffsets.get(expert);
        int assignmentEnd = assignmentStart + expertCounts.get(expert);
        float[] partials = context.allocateFloatLocalArray(
                ACTIVE_EXPERT_TOKEN_TILE * localWorkGroupSize);
        int blocksPerRow = (moeHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = (expert * dim + row) * blocksPerRow;

        for (int tileStart = assignmentStart; tileStart < assignmentEnd;
                tileStart += ACTIVE_EXPERT_TOKEN_TILE) {
            int tokenCount = assignmentEnd - tileStart;
            if (tokenCount > ACTIVE_EXPERT_TOKEN_TILE) {
                tokenCount = ACTIVE_EXPERT_TOKEN_TILE;
            }
            for (int tileToken = 0; tileToken < ACTIVE_EXPERT_TOKEN_TILE; tileToken++) {
                partials[tileToken * localWorkGroupSize + localId] = 0.0f;
            }
            for (int column = localId; column < moeHiddenDim; column += localWorkGroupSize) {
                int blockByteOffset = (rowBlockOffset + column / Q8_0_BLOCK_SIZE)
                        * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;
                float weight = downExperts.get(quantOffset)
                        * downExperts.getHalfFloat(blockByteOffset).getFloat32();
                for (int tileToken = 0; tileToken < tokenCount; tileToken++) {
                    int groupedPosition = tileStart + tileToken;
                    partials[tileToken * localWorkGroupSize + localId]
                            += weight * groupedHidden.get(groupedPosition * moeHiddenDim + column);
                }
            }
            for (int tileToken = 0; tileToken < ACTIVE_EXPERT_TOKEN_TILE; tileToken++) {
                int base = tileToken * localWorkGroupSize;
                reduceLocalAt(context, partials, base, localId, localWorkGroupSize);
                if (localId == 0 && tileToken < tokenCount) {
                    int groupedPosition = tileStart + tileToken;
                    int assignment = groupedAssignmentIds.get(groupedPosition);
                    groupedDown.set(groupedPosition * dim + row,
                            routingWeights.get(assignment) * partials[base]);
                }
            }
        }
    }

    /** Scatters grouped routed outputs back to their original token rows. */
    public static void scatterGroupedRoutedOutput(
            KernelContext context,
            FloatArray groupedDown,
            FloatArray residual,
            IntArray groupedPositionByAssignment,
            IntArray activeBatchSizeHolder,
            int dim,
            int topK) {

        int index = context.globalIdx;
        if (index >= activeBatchSizeHolder.get(0) * dim) {
            return;
        }
        int token = index / dim;
        int row = index % dim;
        float sum = 0.0f;
        for (int slot = 0; slot < topK; slot++) {
            int assignment = token * topK + slot;
            int groupedPosition = groupedPositionByAssignment.get(assignment);
            sum += groupedDown.get(groupedPosition * dim + row);
        }
        residual.set(index, residual.get(index) + sum);
    }

    /** Computes every token's shared-expert SwiGLU hidden vector and gate weight. */
    public static void batchedSharedGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray input,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray sharedHidden,
            FloatArray sharedGateInput,
            FloatArray sharedWeights,
            IntArray activeBatchSizeHolder,
            int dim,
            int sharedHiddenDim,
            int localWorkGroupSize) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int token = groupId / sharedHiddenDim;
        int row = groupId % sharedHiddenDim;
        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = row * blocksPerRow;
        int inputOffset = token * dim;
        float[] gateBlockScales = context.allocateFloatLocalArray(blocksPerRow);
        float[] upBlockScales = context.allocateFloatLocalArray(blocksPerRow);
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * Q8_0_BLOCK_BYTES;
            gateBlockScales[block] = sharedGate.getHalfFloat(blockByteOffset).getFloat32();
            upBlockScales[block] = sharedUp.getHalfFloat(blockByteOffset).getFloat32();
        }
        context.localBarrier();
        float gatePartial = 0.0f;
        float upPartial = 0.0f;
        float sharedScorePartial = 0.0f;
        for (int column = localId; column < dim; column += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
            int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;
            float inputValue = input.get(inputOffset + column);
            int block = column / Q8_0_BLOCK_SIZE;
            gatePartial += sharedGate.get(quantOffset) * gateBlockScales[block] * inputValue;
            upPartial += sharedUp.get(quantOffset) * upBlockScales[block] * inputValue;
            if (row == 0) {
                sharedScorePartial += sharedGateInput.get(column) * inputValue;
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = gatePartial;
        context.localBarrier();
        reduceLocal(context, localSums, localId, localWorkGroupSize);
        float gate = localSums[0];
        context.localBarrier();

        localSums[localId] = upPartial;
        context.localBarrier();
        reduceLocal(context, localSums, localId, localWorkGroupSize);
        if (localId == 0) {
            float siluGate = gate / (1.0f + TornadoMath.exp(-gate));
            sharedHidden.set(token * sharedHiddenDim + row, siluGate * localSums[0]);
        }

        // Do not reuse local memory until thread zero has consumed the Up sum.
        context.localBarrier();
        if (row == 0) {
            localSums[localId] = sharedScorePartial;
            context.localBarrier();
            reduceLocal(context, localSums, localId, localWorkGroupSize);
            if (localId == 0) {
                float score = localSums[0];
                sharedWeights.set(token, 1.0f / (1.0f + TornadoMath.exp(-score)));
            }
        }
    }

    /** Down-projects the shared expert and adds its gated output to each token. */
    public static void batchedSharedDownAndAccumulateQ8_0(
            KernelContext context,
            FloatArray sharedHidden,
            ByteArray sharedDown,
            FloatArray sharedWeights,
            FloatArray residual,
            IntArray activeBatchSizeHolder,
            int dim,
            int sharedHiddenDim,
            int localWorkGroupSize) {

        int groupId = context.groupIdx;
        int localId = context.localIdx;
        int token = groupId / dim;
        int row = groupId % dim;
        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int blocksPerRow = (sharedHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = row * blocksPerRow;
        int hiddenOffset = token * sharedHiddenDim;
        float[] downBlockScales = context.allocateFloatLocalArray(blocksPerRow);
        for (int block = localId; block < blocksPerRow; block += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + block) * Q8_0_BLOCK_BYTES;
            downBlockScales[block] = sharedDown.getHalfFloat(blockByteOffset).getFloat32();
        }
        context.localBarrier();
        float partial = 0.0f;
        for (int column = localId; column < sharedHiddenDim; column += localWorkGroupSize) {
            int blockByteOffset = (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
            int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;
            float weight = sharedDown.get(quantOffset)
                    * downBlockScales[column / Q8_0_BLOCK_SIZE];
            partial += weight * sharedHidden.get(hiddenOffset + column);
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partial;
        context.localBarrier();
        reduceLocal(context, localSums, localId, localWorkGroupSize);
        if (localId == 0) {
            int output = token * dim + row;
            residual.set(output, residual.get(output) + sharedWeights.get(token) * localSums[0]);
        }
    }

    private static void reduceLocal(
            KernelContext context,
            float[] localSums,
            int localId,
            int localWorkGroupSize) {
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
    }

    private static void reduceLocalAt(
            KernelContext context,
            float[] localSums,
            int baseOffset,
            int localId,
            int localWorkGroupSize) {
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[baseOffset + localId] += localSums[baseOffset + localId + stride];
            }
            context.localBarrier();
        }
    }
}
