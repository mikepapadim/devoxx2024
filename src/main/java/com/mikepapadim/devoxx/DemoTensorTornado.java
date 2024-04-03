package com.mikepapadim.devoxx;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;
import uk.ac.manchester.tornado.api.types.tensors.Shape;
import uk.ac.manchester.tornado.api.types.tensors.TensorFloat32;

public class DemoTensorTornado {
    public static void tensorAdditionFloat32(TensorFloat32 tensorA, TensorFloat32 tensorB, TensorFloat32 tensorC) {
        for (@Parallel int i = 0; i < tensorC.getSize(); i++) {
            tensorC.set(i, tensorA.get(i) + tensorB.get(i));
        }
    }

    public static void main(String[] args) {
        // Define the shape for the tensors
        Shape shape = new Shape(16, 16);

        // Create two tensors and initialize their values
        TensorFloat32 tensorA = new TensorFloat32(shape);
        TensorFloat32 tensorB = new TensorFloat32(shape);

        // Create a tensor to store the result of addition
        TensorFloat32 tensorC = new TensorFloat32(shape);

        // Initialize tensors
        tensorA.init(20f);
        tensorB.init(300f);

        // Define the task graph
        TaskGraph taskGraph = new TaskGraph("s0")
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, tensorA, tensorB)
                .task("t0", DemoTensorTornado::tensorAdditionFloat32, tensorA, tensorB, tensorC)
                .transferToHost(DataTransferMode.EVERY_EXECUTION, tensorC);

        // Take a snapshot of the task graph
        ImmutableTaskGraph immutableTaskGraph = taskGraph.snapshot();

        // Create an execution plan and execute it
        TornadoExecutionPlan executionPlan = new TornadoExecutionPlan(immutableTaskGraph);
        executionPlan.execute();

        prettyPrintTensor(tensorC, tensorC.getShape());
    }

    public static void prettyPrintTensor(TensorFloat32 tensor, Shape shape) {
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

}
