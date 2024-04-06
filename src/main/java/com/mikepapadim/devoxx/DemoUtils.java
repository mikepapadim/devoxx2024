package com.mikepapadim.devoxx;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import uk.ac.manchester.tornado.api.types.tensors.Shape;
import uk.ac.manchester.tornado.api.types.tensors.TensorFP32;

public class DemoUtils {
    static void printFormatted(float[] array) {
        for (float value : array) {
            System.out.format("%.2f ", value);
        }
        System.out.println(); // New line for clarity.
    }
    public static void prettyPrintTensor(TensorFP32 tensor, Shape shape) {
        long[] dimensions = shape.getDimensions();
        // Ensure the method is compatible with 2D tensors
        if (dimensions.length != 2) {
            throw new IllegalArgumentException("This pretty print method is designed for 2D tensors.");
        }
        System.out.println("Tensor C Contents:");
        for (int i = 0; i < dimensions[0]; i++) { // Iterate over the first dimension (rows)
            for (int j = 0; j < dimensions[1]; j++) { // Iterate over the second dimension (columns)
                // Calculate the linear index for the current element (row-major order)
                int index = (int) (i * dimensions[1] + j);
                System.out.printf("%.2f ", tensor.get(index));
            }
            System.out.println(); // New line at the end of each row for readability
        }
    }

    static void processOutput(OrtSession.Result outputMap, String tensorOuput) {
        outputMap.get(tensorOuput).ifPresentOrElse(
                tensor -> {
                    // Here you can add code to process the tensor, e.g., extract data, print values, etc.
                    System.out.println("Output tensor received and processed.\n"+ tensor.toString());
                    System.out.println(tensor.getInfo());
                    try {
                        System.out.println(tensor.getValue());
                    } catch (OrtException e) {
                        throw new RuntimeException(e);
                    }
                },
                () -> System.err.println("Output tensor not found in model output.")
        );
    }

    static void printConcatenatedTensorContents(TensorFP32 tensor, int elementsPerTensor) {
        float[] heapArray = tensor.toHeapArray();
        for (int i = 0; i < heapArray.length; i++) {
            System.out.printf("%.2f ", heapArray[i]);
            if ((i + 1) % elementsPerTensor == 0) {
                System.out.println(); // New line to visually separate concatenated tensors.
            }
        }
        System.out.println(); // Extra space for clarity.
    }

     static void printTensorContents(TensorFP32 tensor, Shape shape) {
        for (int i = 0; i < shape.getSize(); i++) {
            System.out.printf("%.2f ", tensor.get(i));
            if ((i + 1) % shape.getDimensions()[1] == 0) {
                System.out.println(); // New line after each row for readability.
            }
        }
        System.out.println(); // Extra space for better separation.
    }
}

