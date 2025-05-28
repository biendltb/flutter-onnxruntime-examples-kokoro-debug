import 'package:flutter/material.dart';
import 'dart:async';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'dart:typed_data';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img; // Using image package for image processing
import 'dart:math' as math;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Classification Demo',
      theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue)),
      home: const OnnxModelDemoPage(title: 'Image Classification Demo'),
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
  final assetPath2 = 'assets/models/kokoro-v1.0.onnx';
  List<Map<String, dynamic>> _displayResults = [];
  List<OrtProvider> _availableProviders = [];
  String? _selectedProvider;
  // Cache for decoded image to avoid decoding it multiple times
  img.Image? _cachedImage;

  @override
  void initState() {
    super.initState();
    _getModelInfo();
  }

  Future<void> _getModelInfo() async {
    _session ??= await OnnxRuntime().createSessionFromAsset(assetPath2);

    // optional: get and set the execution provider
    _availableProviders = await OnnxRuntime().getAvailableProviders();
    setState(() {
      _selectedProvider = _availableProviders.isNotEmpty ? _availableProviders[0].name : null;
    });

    final modelMetadata = await _session!.getMetadata();
    final modelMetadataMap = modelMetadata.toMap();
    final List<Map<String, dynamic>> modelInputInfoMap = await _session!.getInputInfo();
    final List<Map<String, dynamic>> modelOutputInfoMap = await _session!.getOutputInfo();

    // generate a list of maps from the modelMetadataMap if the values is not empty
    final displayList = [
      {'title': 'Model Name', 'value': assetPath2.split('/').last},
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

    OrtProvider provider;
    if (_selectedProvider == null) {
      provider = OrtProvider.CPU;
    } else {
      provider = OrtProvider.values.firstWhere((p) => p.name == _selectedProvider);
    }

    final sessionOptions = OrtSessionOptions(providers: [provider]);

    _session ??= await OnnxRuntime().createSessionFromAsset(assetPath2, options: sessionOptions);

    // read image data and run inference
    _session ??= await OnnxRuntime().createSessionFromAsset(assetPath2);


    final inputs = {
      'tokens': await OrtValue.fromList(Int64List.fromList([0, 50, 156, 43, 102, 16, 44, 123, 156, 138, 81, 83, 123, 0]), [1, 14]),
      'style': await OrtValue.fromList(Float32List.fromList(List.generate(256, (index) => index.toDouble() / 256)), [1, 256]),
      'speed': await OrtValue.fromList(Float32List.fromList([1.0]), [1]),
    };
  
    // RestNet18 has only one input and one output so we just get the first one in the lists
    final String inputName = _session!.inputNames.first;
    final String outputName = _session!.outputNames.first;

    // Run inference
    final startTime = DateTime.now();
    final outputs = await _session!.run(inputs);
    final endTime = DateTime.now();

    // Get the results
    // Resnet18 returns a float32 list, we cast it to a list of doubles since Dart doesn't support float32
    final List<double> audioData = (await outputs[outputName]!.asFlattenedList()).cast<double>();

    // Calculate inference time
    final inferenceTime = endTime.difference(startTime).inMilliseconds;

    // Get the model file size
    final asset = await rootBundle.load(assetPath2);
    final modelSizeInMB = (asset.lengthInBytes / (1024 * 1024)).toStringAsFixed(1);

    // Clean up resources
    for (var output in outputs.values) {
      await output.dispose();
    }

    // Update results
    setState(() {
      _displayResults = [
        {'title': 'Model Name', 'value': assetPath2.split('/').last},
        {'title': 'Model Size', 'value': '$modelSizeInMB MB'},
        {'title': 'Output Data', 'value': audioData.sublist(audioData.length - 100).toString()}, // display last 100 values
        {'title': 'Inference Time', 'value': '$inferenceTime ms'},
        {'title': 'Processing Device', 'value': _selectedProvider ?? 'CPU'},
      ];
      _isProcessing = false;
    });
  }

  List<double> _applySoftmax(List<double> logits) {
    // Find the maximum value to avoid numerical instability
    double maxLogit = logits.reduce((curr, next) => curr > next ? curr : next);

    // Subtract max from each value for numerical stability
    List<double> expValues = logits.map((logit) => math.exp(logit - maxLogit)).toList();

    // Calculate sum of all exp values
    double sumExp = expValues.reduce((sum, val) => sum + val);

    // Normalize by dividing each by the sum
    return expValues.map((expVal) => expVal / sumExp).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(backgroundColor: Theme.of(context).colorScheme.inversePrimary, title: Text(widget.title)),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Dropdown for selecting execution provider
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                children: [
                  const Text('Provider:', style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold)),
                  const SizedBox(width: 20),
                  DropdownButton<String>(
                    value: _selectedProvider,
                    hint: const Text('Select Execution Provider'),
                    items:
                        _availableProviders.map((provider) {
                          return DropdownMenuItem<String>(value: provider.name, child: Text(provider.name));
                        }).toList(),
                    onChanged: (value) {
                      setState(() {
                        _selectedProvider = value;
                      });
                    },
                  ),
                ],
              ),
            ),
            // Get Model Info and Predict buttons
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
