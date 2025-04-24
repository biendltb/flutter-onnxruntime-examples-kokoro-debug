# flutter_onnxruntime_examples

Examples for [ONNX Runtime Flutter](https://github.com/masicai/flutter_onnxruntime)

## Examples

### Image Classification Example

This example demonstrates how to use the ONNX Runtime Flutter plugin for image classification tasks. It includes loading a pre-trained model, preparing input data, and running inference.


<div style="display: flex; justify-content: space-between;">
    <img src="images/classification_metadata.png" width="300" alt="Image classification model metadata"/>
    <img src="images/classification_inference.png" width="300" alt="Image classification inference results"/>
    <img src="images/classification_inference_web.png" width="300" alt="Image classification inference results on web"/>
</div>

To run this example, follow these steps:

* Download the Resnet18 model from ONNX Model Zoo: [Resnet18](https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx)
* Place the model file in the `assets/models` directory.
* Run the app:
    ```bash
    flutter run
    ```
