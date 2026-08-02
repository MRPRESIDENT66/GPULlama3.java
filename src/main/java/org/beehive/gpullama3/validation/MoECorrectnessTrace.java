package org.beehive.gpullama3.validation;

import org.beehive.gpullama3.tensor.standard.FloatTensor;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Optional JSONL trace used to compare Qwen2-MoE CPU and GPU inference.
 *
 * <p>The trace is disabled unless {@code -Dllama.correctnessTrace=/path/file.jsonl}
 * is supplied, so normal inference does not perform any file I/O.</p>
 */
public final class MoECorrectnessTrace {

    private static final String TRACE_PATH = System.getProperty("llama.correctnessTrace");
    private static final boolean ENABLED = TRACE_PATH != null && !TRACE_PATH.isBlank();
    private static final AtomicInteger TOKEN_INDEX = new AtomicInteger();
    private static final BufferedWriter WRITER = createWriter();

    private MoECorrectnessTrace() {
    }

    public static boolean isEnabled() {
        return ENABLED;
    }

    public static void recordCpuRouter(int position, int layer, float[] logits,
                                       int[] experts, float[] weights) {
        if (!ENABLED) {
            return;
        }
        writeRouterPrefix(position, layer);
        writeFloatArray(logits);
        write(",\"experts\":");
        writeIntArray(experts);
        write(",\"weights\":");
        writeFloatArray(weights);
        writeLine("}");
    }

    public static void recordGpuRouter(int position, int layer, FloatArray logits,
                                       IntArray experts, FloatArray weights) {
        if (!ENABLED) {
            return;
        }
        writeRouterPrefix(position, layer);
        writeFloatArray(logits);
        write(",\"experts\":");
        writeIntArray(experts);
        write(",\"weights\":");
        writeFloatArray(weights);
        writeLine("}");
    }

    public static void recordCpuLogits(int position, FloatTensor logits) {
        if (!ENABLED) {
            return;
        }
        write("{\"type\":\"logits\",\"position\":" + position + ",\"values\":");
        writeFloatTensor(logits);
        writeLine("}");
    }

    public static void recordGpuLogits(int position, FloatArray logits) {
        if (!ENABLED) {
            return;
        }
        write("{\"type\":\"logits\",\"position\":" + position + ",\"values\":");
        writeFloatArray(logits);
        writeLine("}");
    }

    public static void recordToken(int tokenId) {
        if (!ENABLED) {
            return;
        }
        writeLine("{\"type\":\"token\",\"index\":" + TOKEN_INDEX.getAndIncrement()
                + ",\"id\":" + tokenId + "}");
    }

    private static BufferedWriter createWriter() {
        if (!ENABLED) {
            return null;
        }
        try {
            Path path = Path.of(TRACE_PATH);
            Path parent = path.toAbsolutePath().getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            BufferedWriter writer = Files.newBufferedWriter(path,
                    StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    writer.close();
                } catch (IOException ignored) {
                    // The process is already terminating.
                }
            }));
            return writer;
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot create correctness trace: " + TRACE_PATH, e);
        }
    }

    private static void writeRouterPrefix(int position, int layer) {
        write("{\"type\":\"router\",\"position\":" + position
                + ",\"layer\":" + layer + ",\"logits\":");
    }

    private static void writeFloatTensor(FloatTensor values) {
        write("[");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) {
                write(",");
            }
            write(Float.toString(values.getFloat(i)));
        }
        write("]");
    }

    private static void writeFloatArray(FloatArray values) {
        write("[");
        for (int i = 0; i < values.getSize(); i++) {
            if (i > 0) {
                write(",");
            }
            write(Float.toString(values.get(i)));
        }
        write("]");
    }

    private static void writeFloatArray(float[] values) {
        write("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                write(",");
            }
            write(Float.toString(values[i]));
        }
        write("]");
    }

    private static void writeIntArray(IntArray values) {
        write("[");
        for (int i = 0; i < values.getSize(); i++) {
            if (i > 0) {
                write(",");
            }
            write(Integer.toString(values.get(i)));
        }
        write("]");
    }

    private static void writeIntArray(int[] values) {
        write("[");
        for (int i = 0; i < values.length; i++) {
            if (i > 0) {
                write(",");
            }
            write(Integer.toString(values[i]));
        }
        write("]");
    }

    private static synchronized void write(String value) {
        try {
            WRITER.write(value);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static synchronized void writeLine(String value) {
        try {
            WRITER.write(value);
            WRITER.newLine();
            WRITER.flush();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
