package com.mikepapadim.devoxx;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.tensors.Shape;
import uk.ac.manchester.tornado.api.types.tensors.TensorFloat32;

public class DemoTensorAPIConcat {
    public static void main(String[] args) {
        // Define the shapes for two 2x2 tensors.
        Shape shape2x2 = new Shape(2, 2);

        // Create and initialize two TensorFloat32 instances with specific values.
        TensorFloat32 tensor1 = createAndFillTensor(shape2x2, 1.0f); // Filled with 1.0
        TensorFloat32 tensor2 = createAndFillTensor(shape2x2, 2.0f); // Filled with 2.0

        // Concatenate tensor1 and tensor2 along a new dimension, resulting in a 2x2x2 tensor.
        // For simplicity in this example, we simulate concatenation by first converting to arrays then joining.
        // Note: Actual 'concat' implementation will depend on how it's defined to work with TensorFloat32 instances.
        FloatArray concatenatedArray = TensorFloat32.concat(tensor1, tensor2);

        // Print the contents of the concatenated tensor.
        System.out.println("Concatenated Tensor Contents:");
        printConcatenatedTensorContents(concatenatedArray, 4); // Assuming the concat adds to the first dimension.
    }

    private static TensorFloat32 createAndFillTensor(Shape shape, float fillValue) {
        TensorFloat32 tensor = new TensorFloat32(shape);
        tensor.init(fillValue);
        return tensor;
    }

    private static void printConcatenatedTensorContents(FloatArray tensor, int elementsPerTensor) {
        float[] heapArray = tensor.toHeapArray();
        for (int i = 0; i < heapArray.length; i++) {
            System.out.printf("%.2f ", heapArray[i]);
            if ((i + 1) % elementsPerTensor == 0) {
                System.out.println(); // New line to visually separate concatenated tensors.
            }
        }
        System.out.println(); // Extra space for clarity.
    }
}
