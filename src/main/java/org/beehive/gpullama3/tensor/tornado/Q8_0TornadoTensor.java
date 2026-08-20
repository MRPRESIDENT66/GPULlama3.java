package org.beehive.gpullama3.tensor.tornado;

import org.beehive.gpullama3.tensor.GGMLType;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;

import java.lang.foreign.MemorySegment;

/**
 * This class represents a quantized tensor in the {@link GGMLType#Q8_0} format.
 * It is backed by a {@link ByteArray} containing both the quantized values and the scale factors.
 * The underlying {@link ByteArray} contains N Q8_0 blocks, where N is the tensor size divided by 32.:
 * Each Q8_0 Block has the following layout:
 * [Scale Factor (fp16) - 2 bytes] [Quantized Value 0 (int8) - 1 byte] ... [Quantized Value 31 (int8) - 1 byte]
 */
public class Q8_0TornadoTensor extends TornadoTensor {

    private static final int BLOCK_SIZE = 32;
    private static final int BLOCK_BYTES = 34;
    private static final int QUANT_ALIGNMENT = 128;

    private final ByteArray tornadoNativeArray; // Unified Q8_0 tensor in the memorySegment of the ByteArray
    private ByteArray repackedArray;

    public Q8_0TornadoTensor(ByteArray byteArray) {
        this.tornadoNativeArray = byteArray;
    }

    public static Q8_0TornadoTensor fromTornadoMemorySegment(MemorySegment segment) {
        return new Q8_0TornadoTensor(ByteArray.fromSegmentShallow(segment));
    }

    @Override
    public ByteArray asByteArray() {
        return tornadoNativeArray;
    }

    @Override
    public ByteArray asRepackedByteArray() {
        ensureRepacked();
        return repackedArray;
    }

    private synchronized void ensureRepacked() {
        if (repackedArray != null) {
            return;
        }

        int numberOfBlocks = tornadoNativeArray.getSize() / BLOCK_BYTES;
        int scaleBytes = numberOfBlocks * 2;
        int quantOffset =
                ((scaleBytes + QUANT_ALIGNMENT - 1) / QUANT_ALIGNMENT) * QUANT_ALIGNMENT;
        ByteArray repacked = new ByteArray(quantOffset + numberOfBlocks * BLOCK_SIZE);

        for (int block = 0; block < numberOfBlocks; block++) {
            int sourceOffset = block * BLOCK_BYTES;
            repacked.setHalfFloat(block * 2, tornadoNativeArray.getHalfFloat(sourceOffset));
            int destinationQuantOffset = quantOffset + block * BLOCK_SIZE;
            for (int element = 0; element < BLOCK_SIZE; element++) {
                repacked.set(
                        destinationQuantOffset + element,
                        tornadoNativeArray.get(sourceOffset + 2 + element));
            }
        }

        repackedArray = repacked;
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

}
