package org.beehive.gpullama3.diagnostics;

import org.beehive.gpullama3.tornadovm.kernels.Qwen2MoEBatchKernels;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.WorkerGrid1D;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.HalfFloat;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/** Temporary diagnostic for comparing the FP32-activation and W8A8 Gate/Up kernels. */
public final class W8A8KernelDiagnostic {

    private static final int DIM = 32;
    private static final int HIDDEN_DIM = 4;
    private static final int Q8_BLOCK_BYTES = 34;
    private static final int LOCAL_SIZE = 128;

    private W8A8KernelDiagnostic() {}

    public static void main(String[] args) throws Exception {
        FloatArray input = new FloatArray(DIM);
        ByteArray quantizedInput = new ByteArray(DIM);
        HalfFloatArray inputScales = new HalfFloatArray(1);
        IntArray activeBatchSize = scalarInt(1);
        IntArray groupedAssignments = scalarInt(0);
        IntArray tileExpertIds = scalarInt(0);
        IntArray tileStarts = scalarInt(0);
        IntArray tileCounts = scalarInt(1);
        IntArray tileCountHolder = scalarInt(1);
        ByteArray gateWeights = q8Weights(3);
        ByteArray upWeights = q8Weights(7);
        FloatArray fp32Output = new FloatArray(HIDDEN_DIM);
        FloatArray w8a8Output = new FloatArray(HIDDEN_DIM);

        for (int i = 0; i < DIM; i++) {
            input.set(i, ((i % 11) - 5) * 0.17f + i * 0.003f);
        }

        WorkerGrid1D quantizeGrid = new WorkerGrid1D(DIM);
        quantizeGrid.setLocalWork(32, 1, 1);
        WorkerGrid1D fp32Grid = new WorkerGrid1D(LOCAL_SIZE);
        fp32Grid.setLocalWork(LOCAL_SIZE, 1, 1);
        WorkerGrid1D w8a8Grid = new WorkerGrid1D(LOCAL_SIZE);
        w8a8Grid.setLocalWork(LOCAL_SIZE, 1, 1);

        GridScheduler scheduler = new GridScheduler();
        scheduler.addWorkerGrid("w8a8Diagnostic.quantize", quantizeGrid);
        scheduler.addWorkerGrid("w8a8Diagnostic.fp32", fp32Grid);
        scheduler.addWorkerGrid("w8a8Diagnostic.w8a8", w8a8Grid);

        TaskGraph graph = new TaskGraph("w8a8Diagnostic")
                .transferToDevice(
                        DataTransferMode.EVERY_EXECUTION,
                        input,
                        activeBatchSize,
                        groupedAssignments,
                        tileExpertIds,
                        tileStarts,
                        tileCounts,
                        tileCountHolder,
                        gateWeights,
                        upWeights)
                .task(
                        "quantize",
                        Qwen2MoEBatchKernels::quantizeBatchFfnInputQ8_0,
                        new KernelContext(),
                        input,
                        quantizedInput,
                        inputScales,
                        activeBatchSize,
                        DIM)
                .task(
                        "fp32",
                        Qwen2MoEBatchKernels::tiled2DRoutedExpertsGateUpSwiGLUQ8_0,
                        new KernelContext(),
                        input,
                        groupedAssignments,
                        tileExpertIds,
                        tileStarts,
                        tileCounts,
                        tileCountHolder,
                        gateWeights,
                        upWeights,
                        fp32Output,
                        DIM,
                        HIDDEN_DIM,
                        1,
                        LOCAL_SIZE)
                .task(
                        "w8a8",
                        Qwen2MoEBatchKernels::tiled2DRoutedExpertsGateUpSwiGLUQ8_0Int8Activation,
                        new KernelContext(),
                        quantizedInput,
                        inputScales,
                        groupedAssignments,
                        tileExpertIds,
                        tileStarts,
                        tileCounts,
                        tileCountHolder,
                        gateWeights,
                        upWeights,
                        w8a8Output,
                        DIM,
                        HIDDEN_DIM,
                        1,
                        LOCAL_SIZE)
                .transferToHost(
                        DataTransferMode.EVERY_EXECUTION,
                        quantizedInput,
                        inputScales,
                        fp32Output,
                        w8a8Output);

        try (TornadoExecutionPlan plan = new TornadoExecutionPlan(graph.snapshot())) {
            plan.withGridScheduler(scheduler).execute();
        }

        printQuantization(input, quantizedInput, inputScales);
        for (int row = 0; row < HIDDEN_DIM; row++) {
            float fp32 = fp32Output.get(row);
            float w8a8 = w8a8Output.get(row);
            System.out.printf(
                    "row=%d fp32=% .8f w8a8=% .8f absError=% .8f%n",
                    row, fp32, w8a8, Math.abs(fp32 - w8a8));
        }
    }

    private static IntArray scalarInt(int value) {
        IntArray result = new IntArray(1);
        result.set(0, value);
        return result;
    }

    private static ByteArray q8Weights(int seed) {
        ByteArray weights = new ByteArray(HIDDEN_DIM * Q8_BLOCK_BYTES);
        for (int row = 0; row < HIDDEN_DIM; row++) {
            int blockOffset = row * Q8_BLOCK_BYTES;
            float scale = 0.006f + row * 0.001f;
            weights.setHalfFloat(blockOffset, new HalfFloat(scale));
            for (int column = 0; column < DIM; column++) {
                int quant = ((column * seed + row * 13) % 63) - 31;
                weights.set(blockOffset + 2 + column, (byte) quant);
            }
        }
        return weights;
    }

    private static void printQuantization(
            FloatArray input, ByteArray quantizedInput, HalfFloatArray inputScales) {
        float maxAbs = 0.0f;
        for (int i = 0; i < DIM; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(input.get(i)));
        }
        float fullScale = maxAbs / 127.0f;
        float expectedStoredScale = new HalfFloat(fullScale).getFloat32();
        float actualStoredScale = inputScales.get(0).getFloat32();
        System.out.printf(
                "scale expected=% .9f actual=% .9f absError=% .9f%n",
                expectedStoredScale,
                actualStoredScale,
                Math.abs(expectedStoredScale - actualStoredScale));
        for (int i = 0; i < 16; i++) {
            float scaled = fullScale == 0.0f ? 0.0f : input.get(i) / fullScale;
            int expected = (int) (scaled + (scaled < 0.0f ? -0.5f : 0.5f));
            expected = Math.max(-127, Math.min(127, expected));
            System.out.printf(
                    "q[%02d] expected=%4d actual=%4d input=% .6f%n",
                    i, expected, quantizedInput.get(i), input.get(i));
        }
    }
}
