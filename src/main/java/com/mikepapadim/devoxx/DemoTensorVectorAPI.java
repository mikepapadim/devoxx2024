package com.mikepapadim.devoxx;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import uk.ac.manchester.tornado.api.types.tensors.Shape;
import uk.ac.manchester.tornado.api.types.tensors.TensorFP32;

import java.nio.ByteOrder;
import java.util.stream.IntStream;

public class DemoTensorVectorAPI {

    public static void main(String[] args) {
        // Step 1: Define the shape of the tensors.
        Shape shape = new Shape( 16, 16); // Creating a 16x16 tensor.

        // Step 2: Create two TensorFP32 instances with the specified shape.
        TensorFP32 tensorA = new TensorFP32(shape);
        TensorFP32 tensorB = new TensorFP32(shape);

        // Step 3: Initialize tensorA with the value 1.0f.
        tensorA.init(1.0f);

        // Initialize tensorB with a sequence of values.
        for (int i = 0; i < tensorB.getSize(); i++) {
            tensorB.set(i, i + 1.0f);
        }

        VectorSpecies<Float> vectorSpecies = FloatVector.SPECIES_PREFERRED;

        float[] tensorData = vectorAdd(tensorA, tensorB, vectorSpecies);

        System.out.println("\nTensor Output:");
        printFormatted(tensorData);

    }
    private static float[] vectorAdd(TensorFP32 vector1, TensorFP32 vector2, VectorSpecies<Float> species) {
        float[] result = new float[vector1.getSize()];
        System.out.println("\nVector API");
        System.out.println(species.toString());
        int width = vector1.getSize() / species.length();
        IntStream.range(0, width).parallel().forEach(i -> {
            long offsetIndex = (long) i * species.length() * Float.BYTES;
            FloatVector vec1 = FloatVector.fromMemorySegment(species, vector1.getSegment(), offsetIndex, ByteOrder.nativeOrder());
            FloatVector vec2 = FloatVector.fromMemorySegment(species, vector2.getSegment(), offsetIndex, ByteOrder.nativeOrder());
            FloatVector resultVec = vec1.add(vec2);
            resultVec.intoArray(result, i * species.length());
        });

        return result;
    }

    private static void printFormatted(float[] array) {
        for (float value : array) {
            System.out.format("%.2f ", value);
        }
        System.out.println(); // New line for clarity.
    }
}
