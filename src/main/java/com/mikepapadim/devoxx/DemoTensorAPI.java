package com.mikepapadim.devoxx;

import uk.ac.manchester.tornado.api.types.tensors.Shape;
import uk.ac.manchester.tornado.api.types.tensors.TensorFloat32;

/**
 * Hello world!
 *
 */
public class DemoTensorAPI {
    public static void main(String[] args) {
        // Step 1: Define the shape of the tensors.
        Shape shape = new Shape( 16, 16); // Creating a 2x2 tensor.

        // Step 2: Create two TensorFloat32 instances with the specified shape.
        TensorFloat32 tensorA = new TensorFloat32(shape);
        TensorFloat32 tensorB = new TensorFloat32(shape);

        // Step 3: Initialize tensorA with the value 1.0f.
        tensorA.init(1.0f);

        // Initialize tensorB with a sequence of values.
        for (int i = 0; i < tensorB.getSize(); i++) {
            tensorB.set(i, i + 1.0f); // Setting values from 1.0f, 2.0f, ..., tensorB.getSize().
        }

        // Step 4: Add tensorA and tensorB element-wise to create tensorC.
        TensorFloat32 tensorC = addTensors(tensorA, tensorB);

        // Step 5: Print the contents of all tensors.
        System.out.println("Tensor A Contents:");
        printTensorContents(tensorA, shape);
        System.out.println("Tensor B Contents:");
        printTensorContents(tensorB, shape);
        System.out.println("Tensor C (A+B) Contents:");
        printTensorContents(tensorC, shape);
    }

    private static TensorFloat32 addTensors(TensorFloat32 tensorA, TensorFloat32 tensorB) {
        if (!tensorA.getShape().equals(tensorB.getShape())) {
            throw new IllegalArgumentException("Tensors must have the same shape for element-wise addition.");
        }
        Shape shape = tensorA.getShape();
        TensorFloat32 tensorC = new TensorFloat32(shape); // Result tensor for storing addition results.
        for (int i = 0; i < tensorC.getSize(); i++) {
            float value = tensorA.get(i) + tensorB.get(i);
            tensorC.set(i, value);
        }
        return tensorC;
    }

    private static void printTensorContents(TensorFloat32 tensor, Shape shape) {
        for (int i = 0; i < shape.getSize(); i++) {
            System.out.printf("%.2f ", tensor.get(i));
            if ((i + 1) % shape.getDimensions()[1] == 0) {
                System.out.println(); // New line after each row for readability.
            }
        }
        System.out.println(); // Extra space for better separation.
    }
}
