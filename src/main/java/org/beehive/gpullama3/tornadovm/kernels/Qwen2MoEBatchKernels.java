package org.beehive.gpullama3.tornadovm.kernels;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** GPU kernels used by Qwen2-MoE batch prefill. */
public final class Qwen2MoEBatchKernels {

    private static final int Q8_0_BLOCK_SIZE = 32;
    private static final int Q8_0_BLOCK_BYTES = 34;

    private Qwen2MoEBatchKernels() {}

    /** Computes one router score for every token-expert pair. */
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

        // One work-group handles one (token, expert) pair.
        // For 60 experts, groups 0..59 process token 0, groups 60..119
        // process token 1, and so on.

        int token = groupId / numberOfExperts;
        int expert = groupId % numberOfExperts;
        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int inputOffset = token * dim;
        int weightOffset = expert * dim;

        // Each local thread computes a strided part of the dot product.
        float partialSum = 0.0f;

        for (int column = localId; column < dim; column += localWorkGroupSize) {

            partialSum +=
                    input.get(inputOffset + column) * routerWeights.get(weightOffset + column);
        }

        // Store every thread's partial sum in local memory.
        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        localSums[localId] = partialSum;
        context.localBarrier();

        // Reduce the partial sums inside this work-group.
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {

            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }

            context.localBarrier();
        }

        // Local thread 0 writes routerLogits[token, expert].
        if (localId == 0) {
            int outputOffset = token * numberOfExperts + expert;
            routerLogits.set(outputOffset, localSums[0]);
        }
    }

    /** Adds Qwen2's Q, K, and V biases independently for every active token. */
    public static void batchedQKVBias(
            KernelContext context,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray qBias,
            FloatArray kBias,
            FloatArray vBias,
            IntArray activeBatchSizeHolder,
            int dim,
            int kvDim) {

        int index = context.globalIdx;
        int rowsPerToken = dim + 2 * kvDim;
        int token = index / rowsPerToken;
        int row = index % rowsPerToken;

        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        if (row < dim) {
            int qIndex = token * dim + row;
            qBatch.set(qIndex, qBatch.get(qIndex) + qBias.get(row));
        } else if (row < dim + kvDim) {
            int kRow = row - dim;
            int kIndex = token * kvDim + kRow;
            kBatch.set(kIndex, kBatch.get(kIndex) + kBias.get(kRow));
        } else {
            int vRow = row - dim - kvDim;
            int vIndex = token * kvDim + vRow;
            vBatch.set(vIndex, vBatch.get(vIndex) + vBias.get(vRow));
        }
    }

    /** Applies Qwen2 split-half RoPE and writes active K/V rows to the cache. */
    public static void batchedRopeWithKVCacheQwen2(
            KernelContext context,
            IntArray batchStartPosHolder,
            IntArray activeBatchSizeHolder,
            FloatArray qBatch,
            FloatArray kBatch,
            FloatArray vBatch,
            FloatArray keyCache,
            FloatArray valueCache,
            int kvDim,
            int headSize,
            int layerIndex,
            int contextLength,
            int dim,
            float ropeTheta) {

        int index = context.globalIdx;
        int pairsPerToken = dim / 2;
        int token = index / pairsPerToken;
        int pair = index % pairsPerToken;

        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int position = batchStartPosHolder.get(0) + token;
        int halfHeadSize = headSize / 2;
        int component = pair % halfHeadSize;
        int head = pair / halfHeadSize;

        float frequency = 1.0f / TornadoMath.pow(ropeTheta, 2.0f * component / (float) headSize);
        float angle = position * frequency;
        float cosine = TornadoMath.cos(angle);
        float sine = TornadoMath.sin(angle);

        int qHeadOffset = token * dim + head * headSize;
        float q0 = qBatch.get(qHeadOffset + component);
        float q1 = qBatch.get(qHeadOffset + component + halfHeadSize);
        qBatch.set(qHeadOffset + component, q0 * cosine - q1 * sine);
        qBatch.set(qHeadOffset + component + halfHeadSize, q0 * sine + q1 * cosine);

        if (pair < kvDim / 2) {
            int kvHead = pair / halfHeadSize;
            int kHeadOffset = token * kvDim + kvHead * headSize;
            float k0 = kBatch.get(kHeadOffset + component);
            float k1 = kBatch.get(kHeadOffset + component + halfHeadSize);
            float rotatedK0 = k0 * cosine - k1 * sine;
            float rotatedK1 = k0 * sine + k1 * cosine;

            kBatch.set(kHeadOffset + component, rotatedK0);
            kBatch.set(kHeadOffset + component + halfHeadSize, rotatedK1);

            int cacheOffset =
                    layerIndex * contextLength * kvDim + position * kvDim + kvHead * headSize;
            keyCache.set(cacheOffset + component, rotatedK0);
            keyCache.set(cacheOffset + component + halfHeadSize, rotatedK1);
            valueCache.set(cacheOffset + component, vBatch.get(kHeadOffset + component));
            valueCache.set(
                    cacheOffset + component + halfHeadSize,
                    vBatch.get(kHeadOffset + component + halfHeadSize));
        }
    }

    /** Applies softmax and selects Top-K experts independently for each token. */
    public static void batchedSoftmaxAndTopK(
            KernelContext context,
            FloatArray routerLogits,
            IntArray selectedExperts,
            FloatArray routingWeights,
            IntArray activeBatchSizeHolder,
            int numberOfExperts,
            int topK) {

        // One GPU thread handles the complete routing result for one token.
        int token = context.globalIdx;
        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        int logitsOffset = token * numberOfExperts;
        int assignmentOffset = token * topK;

        // Find this token's maximum router logit.
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (int i = logitsOffset; i < logitsOffset + numberOfExperts; i++) {
            maxLogit = Math.max(maxLogit, routerLogits.get(i));
        }

        // Compute the stable softmax denominator.
        float sumExp = 0.0f;

        for (int expert = 0; expert < numberOfExperts; expert++) {
            float logit = routerLogits.get(logitsOffset + expert);
            sumExp += TornadoMath.exp(logit - maxLogit);
        }

        // Convert this token's logits to probabilities.
        for (int expert = 0; expert < numberOfExperts; expert++) {
            int index = logitsOffset + expert;
            float logit = routerLogits.get(index);

            float probability = TornadoMath.exp(logit - maxLogit) / sumExp;

            routerLogits.set(index, probability);
        }

        // Select Top-K expert IDs and their routing weights.
        for (int slot = 0; slot < topK; slot++) {
            float currentMax = Float.NEGATIVE_INFINITY;
            int selectedExpert = -1;

            for (int expert = 0; expert < numberOfExperts; expert++) {
                float probability = routerLogits.get(logitsOffset + expert);

                if (probability > currentMax) {
                    currentMax = probability;
                    selectedExpert = expert;
                }
            }

            selectedExperts.set(assignmentOffset + slot, selectedExpert);
            routingWeights.set(assignmentOffset + slot, currentMax);

            routerLogits.set(logitsOffset + selectedExpert, Float.NEGATIVE_INFINITY);
        }
    }

    /** Groups assignments and builds fixed-size tiles containing one expert's tokens. */
    public static void groupAssignmentsAndBuildExpertTiles(
            KernelContext context,
            IntArray selectedExperts,
            IntArray groupedAssignmentIds,
            IntArray groupedPositionByAssignment,
            IntArray expertTileIds,
            IntArray expertTileStarts,
            IntArray expertTileCounts,
            IntArray expertTileCountHolder,
            IntArray activeBatchSizeHolder,
            int numberOfExperts,
            int topK,
            int expertTileSize) {

        if (context.globalIdx != 0) {
            return;
        }

        int numberOfAssignments = activeBatchSizeHolder.get(0) * topK;
        int groupedPosition = 0;
        int tilePosition = 0;

        for (int expert = 0; expert < numberOfExperts; expert++) {
            int expertStart = groupedPosition;
            for (int assignment = 0; assignment < numberOfAssignments; assignment++) {
                if (selectedExperts.get(assignment) == expert) {
                    groupedAssignmentIds.set(groupedPosition, assignment);
                    groupedPositionByAssignment.set(assignment, groupedPosition);
                    groupedPosition++;
                }
            }

            for (int tileStart = expertStart;
                    tileStart < groupedPosition;
                    tileStart += expertTileSize) {
                int tileCount = groupedPosition - tileStart;
                if (tileCount > expertTileSize) {
                    tileCount = expertTileSize;
                }
                expertTileIds.set(tilePosition, expert);
                expertTileStarts.set(tilePosition, tileStart);
                expertTileCounts.set(tilePosition, tileCount);
                tilePosition++;
            }
        }

        expertTileCountHolder.set(0, tilePosition);
    }

    /** Computes a 2-token by 4-output-row Gate/Up tile in one work-group. */
    public static void tiled2DRoutedExpertsGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray inputBatch,
            IntArray groupedAssignmentIds,
            IntArray expertTileIds,
            IntArray expertTileStarts,
            IntArray expertTileCounts,
            IntArray expertTileCountHolder,
            ByteArray gateExperts,
            ByteArray upExperts,
            FloatArray groupedExpertHidden,
            int dim,
            int moeHiddenDim,
            int topK,
            int localWorkGroupSize) {

        int localId = context.localIdx;
        int lane = localId % Q8_0_BLOCK_SIZE;
        int rowInTile = localId / Q8_0_BLOCK_SIZE;
        int rowsPerGroup = localWorkGroupSize / Q8_0_BLOCK_SIZE;
        int outputRowTiles = (moeHiddenDim + rowsPerGroup - 1) / rowsPerGroup;
        int tilePosition = context.groupIdx / outputRowTiles;

        if (tilePosition >= expertTileCountHolder.get(0)) {
            return;
        }

        int outputRowTile = context.groupIdx % outputRowTiles;
        int rowId = outputRowTile * rowsPerGroup + rowInTile;
        boolean activeRow = rowId < moeHiddenDim;
        int expert = expertTileIds.get(tilePosition);
        int tileStart = expertTileStarts.get(tilePosition);
        int tileCount = expertTileCounts.get(tilePosition);
        int token0 = groupedAssignmentIds.get(tileStart) / topK;
        int token1 = token0;
        if (tileCount > 1) {
            token1 = groupedAssignmentIds.get(tileStart + 1) / topK;
        }

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        float gatePartial0 = 0.0f;
        float upPartial0 = 0.0f;
        float gatePartial1 = 0.0f;
        float upPartial1 = 0.0f;
        if (activeRow) {
            int rowBlockOffset = (expert * moeHiddenDim + rowId) * blocksPerRow;
            int inputOffset0 = token0 * dim;
            int inputOffset1 = token1 * dim;

            for (int block = 0; block < blocksPerRow; block++) {
                int column = block * Q8_0_BLOCK_SIZE + lane;
                if (column >= dim) {
                    continue;
                }
                int blockByteOffset = (rowBlockOffset + block) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + lane;

                float gateWeight =
                        gateExperts.get(quantOffset)
                                * gateExperts.getHalfFloat(blockByteOffset).getFloat32();
                float upWeight =
                        upExperts.get(quantOffset)
                                * upExperts.getHalfFloat(blockByteOffset).getFloat32();

                float input0 = inputBatch.get(inputOffset0 + column);
                gatePartial0 += gateWeight * input0;
                upPartial0 += upWeight * input0;
                if (tileCount > 1) {
                    float input1 = inputBatch.get(inputOffset1 + column);
                    gatePartial1 += gateWeight * input1;
                    upPartial1 += upWeight * input1;
                }
            }
        }

        float[] localPartials = context.allocateFloatLocalArray(4 * localWorkGroupSize);
        localPartials[localId] = gatePartial0;
        localPartials[localWorkGroupSize + localId] = upPartial0;
        localPartials[2 * localWorkGroupSize + localId] = gatePartial1;
        localPartials[3 * localWorkGroupSize + localId] = upPartial1;
        context.localBarrier();

        int subgroupBase = rowInTile * Q8_0_BLOCK_SIZE;
        for (int stride = Q8_0_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                int target = subgroupBase + lane;
                int source = target + stride;
                localPartials[target] += localPartials[source];
                localPartials[localWorkGroupSize + target] +=
                        localPartials[localWorkGroupSize + source];
                localPartials[2 * localWorkGroupSize + target] +=
                        localPartials[2 * localWorkGroupSize + source];
                localPartials[3 * localWorkGroupSize + target] +=
                        localPartials[3 * localWorkGroupSize + source];
            }
            context.localBarrier();
        }

        if (lane == 0 && activeRow) {
            float gate0 = localPartials[subgroupBase];
            float up0 = localPartials[localWorkGroupSize + subgroupBase];
            float siluGate0 = gate0 / (1.0f + TornadoMath.exp(-gate0));
            groupedExpertHidden.set(tileStart * moeHiddenDim + rowId, siluGate0 * up0);
            if (tileCount > 1) {
                float gate1 = localPartials[2 * localWorkGroupSize + subgroupBase];
                float up1 = localPartials[3 * localWorkGroupSize + subgroupBase];
                float siluGate1 = gate1 / (1.0f + TornadoMath.exp(-gate1));
                groupedExpertHidden.set((tileStart + 1) * moeHiddenDim + rowId, siluGate1 * up1);
            }
        }
    }

    /** Computes a 2-token by 4-output-row Down tile in one work-group. */
    public static void tiled2DRoutedExpertsDownQ8_0(
            KernelContext context,
            FloatArray groupedExpertHidden,
            IntArray expertTileIds,
            IntArray expertTileStarts,
            IntArray expertTileCounts,
            IntArray expertTileCountHolder,
            ByteArray downExperts,
            FloatArray groupedExpertDown,
            int dim,
            int moeHiddenDim,
            int localWorkGroupSize) {

        int localId = context.localIdx;
        int lane = localId % Q8_0_BLOCK_SIZE;
        int rowInTile = localId / Q8_0_BLOCK_SIZE;
        int rowsPerGroup = localWorkGroupSize / Q8_0_BLOCK_SIZE;
        int outputRowTiles = (dim + rowsPerGroup - 1) / rowsPerGroup;
        int tilePosition = context.groupIdx / outputRowTiles;

        if (tilePosition >= expertTileCountHolder.get(0)) {
            return;
        }

        int outputRowTile = context.groupIdx % outputRowTiles;
        int rowId = outputRowTile * rowsPerGroup + rowInTile;
        boolean activeRow = rowId < dim;
        int expert = expertTileIds.get(tilePosition);
        int tileStart = expertTileStarts.get(tilePosition);
        int tileCount = expertTileCounts.get(tilePosition);
        int hiddenOffset0 = tileStart * moeHiddenDim;
        int hiddenOffset1 = (tileStart + 1) * moeHiddenDim;
        int blocksPerRow = (moeHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        float partial0 = 0.0f;
        float partial1 = 0.0f;
        if (activeRow) {
            int rowBlockOffset = (expert * dim + rowId) * blocksPerRow;

            for (int block = 0; block < blocksPerRow; block++) {
                int column = block * Q8_0_BLOCK_SIZE + lane;
                if (column >= moeHiddenDim) {
                    continue;
                }
                int blockByteOffset = (rowBlockOffset + block) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + lane;
                float weight =
                        downExperts.get(quantOffset)
                                * downExperts.getHalfFloat(blockByteOffset).getFloat32();

                partial0 += weight * groupedExpertHidden.get(hiddenOffset0 + column);
                if (tileCount > 1) {
                    partial1 += weight * groupedExpertHidden.get(hiddenOffset1 + column);
                }
            }
        }

        float[] localPartials = context.allocateFloatLocalArray(2 * localWorkGroupSize);
        localPartials[localId] = partial0;
        localPartials[localWorkGroupSize + localId] = partial1;
        context.localBarrier();

        int subgroupBase = rowInTile * Q8_0_BLOCK_SIZE;
        for (int stride = Q8_0_BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                int target = subgroupBase + lane;
                int source = target + stride;
                localPartials[target] += localPartials[source];
                localPartials[localWorkGroupSize + target] +=
                        localPartials[localWorkGroupSize + source];
            }
            context.localBarrier();
        }

        if (lane == 0 && activeRow) {
            groupedExpertDown.set(tileStart * dim + rowId, localPartials[subgroupBase]);
            if (tileCount > 1) {
                groupedExpertDown.set(
                        (tileStart + 1) * dim + rowId,
                        localPartials[localWorkGroupSize + subgroupBase]);
            }
        }
    }

    /** Adds the weighted routed-expert results back to each token's residual. */
    public static void accumulateGroupedRoutedExperts(
            KernelContext context,
            FloatArray groupedExpertDown,
            IntArray groupedPositionByAssignment,
            FloatArray routingWeights,
            FloatArray residualBatch,
            IntArray activeBatchSizeHolder,
            int dim,
            int topK) {

        int index = context.globalIdx;
        int token = index / dim;
        int rowId = index % dim;

        if (token >= activeBatchSizeHolder.get(0)) {
            return;
        }

        float result = residualBatch.get(index);
        int assignmentOffset = token * topK;
        for (int slot = 0; slot < topK; slot++) {
            int assignment = assignmentOffset + slot;
            int groupedPosition = groupedPositionByAssignment.get(assignment);
            int downOffset = groupedPosition * dim + rowId;
            result += routingWeights.get(assignment) * groupedExpertDown.get(downOffset);
        }

        residualBatch.set(index, result);
    }

    /** Computes the shared expert Gate/Up result independently for each token. */
    public static void batchedSharedExpertGateUpSwiGLUQ8_0(
            KernelContext context,
            FloatArray inputBatch,
            IntArray activeBatchSizeHolder,
            ByteArray sharedGate,
            ByteArray sharedUp,
            FloatArray sharedHiddenBatch,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int flatGroupId = context.groupIdx;
        int localId = context.localIdx;

        int token = flatGroupId / sharedExpertHiddenDim;
        int rowId = flatGroupId % sharedExpertHiddenDim;
        boolean active = token < activeBatchSizeHolder.get(0);

        int blocksPerRow = (dim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = rowId * blocksPerRow;
        int inputOffset = token * dim;

        float gatePartialSum = 0.0f;
        float upPartialSum = 0.0f;
        if (active) {
            for (int column = localId; column < dim; column += localWorkGroupSize) {

                int blockByteOffset =
                        (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

                float inputValue = inputBatch.get(inputOffset + column);
                float gateScale = sharedGate.getHalfFloat(blockByteOffset).getFloat32();
                float upScale = sharedUp.getHalfFloat(blockByteOffset).getFloat32();

                gatePartialSum += sharedGate.get(quantOffset) * gateScale * inputValue;
                upPartialSum += sharedUp.get(quantOffset) * upScale * inputValue;
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);

        localSums[localId] = gatePartialSum;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }
        float gate = localSums[0];

        localSums[localId] = upPartialSum;
        context.localBarrier();
        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0 && active) {
            float up = localSums[0];
            float siluGate = gate / (1.0f + TornadoMath.exp(-gate));
            int outputOffset = token * sharedExpertHiddenDim + rowId;
            sharedHiddenBatch.set(outputOffset, siluGate * up);
        }
    }

    /** Computes the sigmoid gate that scales each token's shared-expert output. */
    public static void batchedSharedExpertGateWeight(
            KernelContext context,
            FloatArray inputBatch,
            FloatArray sharedGateInput,
            FloatArray sharedWeightBatch,
            IntArray activeBatchSizeHolder,
            int dim,
            int localWorkGroupSize) {

        int token = context.groupIdx;
        int localId = context.localIdx;
        boolean active = token < activeBatchSizeHolder.get(0);
        int inputOffset = token * dim;

        float partialScore = 0.0f;
        if (active) {
            for (int column = localId; column < dim; column += localWorkGroupSize) {
                partialScore += sharedGateInput.get(column) * inputBatch.get(inputOffset + column);
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partialScore;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0 && active) {
            float sharedWeight = 1.0f / (1.0f + TornadoMath.exp(-localSums[0]));
            sharedWeightBatch.set(token, sharedWeight);
        }
    }

    /** Down-projects the shared expert and adds it to each token's residual. */
    public static void batchedSharedExpertDownAndAccumulateQ8_0(
            KernelContext context,
            FloatArray sharedHiddenBatch,
            FloatArray sharedWeightBatch,
            IntArray activeBatchSizeHolder,
            ByteArray sharedDown,
            FloatArray residualBatch,
            int dim,
            int sharedExpertHiddenDim,
            int localWorkGroupSize) {

        int flatGroupId = context.groupIdx;
        int localId = context.localIdx;

        int token = flatGroupId / dim;
        int rowId = flatGroupId % dim;
        boolean active = token < activeBatchSizeHolder.get(0);

        int blocksPerRow = (sharedExpertHiddenDim + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE;
        int rowBlockOffset = rowId * blocksPerRow;
        int hiddenOffset = token * sharedExpertHiddenDim;

        float partialSum = 0.0f;
        if (active) {
            for (int column = localId;
                    column < sharedExpertHiddenDim;
                    column += localWorkGroupSize) {

                int blockByteOffset =
                        (rowBlockOffset + column / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_BYTES;
                int quantOffset = blockByteOffset + 2 + column % Q8_0_BLOCK_SIZE;

                float weight =
                        sharedDown.get(quantOffset)
                                * sharedDown.getHalfFloat(blockByteOffset).getFloat32();
                partialSum += weight * sharedHiddenBatch.get(hiddenOffset + column);
            }
        }

        float[] localSums = context.allocateFloatLocalArray(localWorkGroupSize);
        localSums[localId] = partialSum;
        context.localBarrier();

        for (int stride = localWorkGroupSize / 2; stride > 0; stride >>= 1) {
            if (localId < stride) {
                localSums[localId] += localSums[localId + stride];
            }
            context.localBarrier();
        }

        if (localId == 0 && active) {
            int outputOffset = token * dim + rowId;
            float weightedOutput = sharedWeightBatch.get(token) * localSums[0];
            residualBatch.set(outputOffset, residualBatch.get(outputOffset) + weightedOutput);
        }
    }
}
