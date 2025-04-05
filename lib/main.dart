import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ONNX Runtime Demo',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue)),
      home: const OnnxModelDemoPage(title: 'ONNX Runtime Demo'),
    );
  }
}

class OnnxModelDemoPage extends StatefulWidget {
  const OnnxModelDemoPage({super.key, required this.title});

  final String title;

  @override
  State<OnnxModelDemoPage> createState() => _OnnxModelDemoPageState();
}

class _OnnxModelDemoPageState extends State<OnnxModelDemoPage> {
  bool _isProcessing = false;
  OrtSession? _session;
  final assetPath = 'assets/models/resnet18-v1-7.onnx';
  List<Map<String, dynamic>> _displayResults = [];

  Future<void> _getModelInfo() async {
    _session ??= await OnnxRuntime().createSessionFromAsset(assetPath);

    final modelMetadata = await _session!.getMetadata();
    final modelMetadataMap = modelMetadata.toMap();
    final List<Map<String, dynamic>> modelInputInfoMap = await _session!.getInputInfo();
    final List<Map<String, dynamic>> modelOutputInfoMap = await _session!.getOutputInfo();

    // generate a list of maps from the modelMetadataMap if the values is not empty
    final displayList = [
      {'title': 'Model Name', 'value': 'ResNet18'},
    ];
    // loop through the list of input and output info and add them to the displayList
    // Add index to the input and output prefix
    for (var i = 0; i < modelInputInfoMap.length; i++) {
      for (var key in modelInputInfoMap[i].keys) {
        displayList.add({'title': 'Input $i: $key', 'value': modelInputInfoMap[i][key].toString()});
      }
    }
    for (var i = 0; i < modelOutputInfoMap.length; i++) {
      for (var key in modelOutputInfoMap[i].keys) {
        displayList.add({'title': 'Output $i: $key', 'value': modelOutputInfoMap[i][key].toString()});
      }
    }

    for (var entry in modelMetadataMap.entries) {
      // if the value is string and empty, skip
      if (entry.value is String && entry.value.isEmpty) {
        continue;
      }
      displayList.add({'title': entry.key, 'value': entry.value.toString()});
    }

    setState(() {
      _displayResults = displayList;
    });
  }

  // Placeholder method to run inference
  Future<void> _runInference() async {
    setState(() {
      _isProcessing = true;
    });

    // Simulate processing time
    await Future.delayed(const Duration(seconds: 1));

    _session ??= await OnnxRuntime().createSessionFromAsset(assetPath);

    // Sample prediction results - this would come from the actual model inference
    setState(() {
      _displayResults = [
        {'title': 'Model Name', 'value': 'ResNet18'},
        {'title': 'Input Name', 'value': 'data'},
        {'title': 'Input Shape', 'value': '[1, 3, 224, 224]'},
        {'title': 'Output Name', 'value': 'resnetv17_dense0_fwd'},
        {'title': 'Top Prediction', 'value': 'Persian cat (Class 285)'},
        {'title': 'Confidence', 'value': '0.92'},
        {'title': 'Inference Time', 'value': '230ms'},
        {'title': 'Model Size', 'value': '44.7MB'},
        {'title': 'Processing Device', 'value': 'CPU'},
      ];
      _isProcessing = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(backgroundColor: Theme.of(context).colorScheme.inversePrimary, title: Text(widget.title)),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Cat image at the top
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 20.0),
              child: Image.asset('assets/images/cat.jpg', height: 200, fit: BoxFit.contain),
            ),

            // Predict button
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(onPressed: _getModelInfo, child: const Text('Get Model Info')),
                const SizedBox(width: 10),
                ElevatedButton(
                  onPressed: _isProcessing ? null : _runInference,
                  style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 12)),
                  child:
                      _isProcessing
                          ? const Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2.0)),
                              SizedBox(width: 12),
                              Text('Processing...'),
                            ],
                          )
                          : const Text('Predict', style: TextStyle(fontSize: 16)),
                ),
              ],
            ),

            const SizedBox(height: 20),

            // Results section
            Expanded(
              child:
                  _displayResults.isEmpty
                      ? const Center(
                        child: Text(
                          'Press the Predict button to run inference',
                          style: TextStyle(fontSize: 16, color: Colors.grey),
                        ),
                      )
                      : ListView.builder(
                        itemCount: _displayResults.length,
                        itemBuilder: (context, index) {
                          final result = _displayResults[index];
                          return Card(
                            margin: const EdgeInsets.only(bottom: 8),
                            child: Padding(
                              padding: const EdgeInsets.all(12.0),
                              child: Row(
                                children: [
                                  Expanded(
                                    flex: 2,
                                    child: Text(
                                      result['title'],
                                      style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                                    ),
                                  ),
                                  Expanded(flex: 3, child: Text(result['value'], style: const TextStyle(fontSize: 14))),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
            ),
          ],
        ),
      ),
    );
  }
}
