package com.mikepapadim.devoxx;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import uk.ac.manchester.tornado.api.types.tensors.Shape;
import uk.ac.manchester.tornado.api.types.tensors.TensorFP32;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class DemoTensorAPIOnnxRT {
    private static final String INPUT_TENSOR_NAME = "data";
    private static final String OUTPUT_TENSOR_NAME = "mobilenetv20_output_flatten0_reshape0";
    private static final String MODEL_PATH = "models/mobilenetv2-7.onnx";

    public static void main(String[] args) {
        try {
            runInference();
        } catch (OrtException e) {
            System.err.println("An error occurred during ONNX Runtime operations: " + e.getMessage());
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("A general error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void runInference() throws OrtException {
        Shape shape = new Shape(1, 3, 224, 224);
        TensorFP32 tornadoTensor = new TensorFP32(shape);
        tornadoTensor.init(2f);

        // Try-with-resources to ensure proper resource management
        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
                OrtSession session = env.createSession(MODEL_PATH, new OrtSession.SessionOptions())) {

            FloatBuffer floatBuffer = tornadoTensor.getFloatBuffer();
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape.dimensions());
            Map<String, OnnxTensor> inputMap = new HashMap<>();
            inputMap.put(INPUT_TENSOR_NAME, inputTensor);

            // Run the model inference and process output
            try (OrtSession.Result result = session.run(inputMap)) {
                processOutput(result);
            } finally {
                inputTensor.close(); // Ensure the input tensor is closed to release resources
            }
        } // Environment and session are auto-closed
    }

    private static void processOutput(OrtSession.Result outputMap) {
        outputMap.get(OUTPUT_TENSOR_NAME).ifPresentOrElse(
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

}
