import 'dart:io';

import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:yolov4tflite/yolov4tflite.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String _platformVersion = 'Unknown';
  bool _busy;

  @override
  void initState() {
    super.initState();
    initPlatformState();
    _busy = false;
  }

  // Platform messages are asynchronous, so we initialize in an async method.
  Future<void> initPlatformState() async {
    String platformVersion;
    // Platform messages may fail, so we use a try/catch PlatformException.
    try {
      platformVersion = await Yolov4tflite.platformVersion;
    } on PlatformException {
      platformVersion = 'Failed to get platform version.';
    }

    // If the widget was removed from the tree while the asynchronous platform
    // message was in flight, we want to discard the reply rather than calling
    // setState to update our non-existent appearance.
    if (!mounted) return;

    setState(() {
      _platformVersion = platformVersion;
    });
  }

  Future openImagePicker() async {
    var image = await ImagePicker.pickImage(source: ImageSource.gallery);
    if (image == null) return;
    setState(() {
      _busy = true;
    });
    predictImage(image);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text('Plugin example app'),
        ),
        body: Center(
          child: Text('Running on: $_platformVersion\n'),
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: openImagePicker,
          tooltip: 'Pick Image',
          child: Icon(Icons.image),
        ),
      ),
    );
  }

  void predictImage(File image) async {
    String path = image.path;

    String labels = await getFileData("assets/labels.txt");
    labels = labels.split("\n").join(",");

    await Yolov4tflite.loadModel(
        modelPath: "assets/yolov4-416-COCO.tflite", labels: labels);
    var r = await Yolov4tflite.detectObjects(imagePath: path);
    print(r);
    setState(() {
      _busy = false;
    });
  }

  Future<String> getFileData(String path) async {
    return await rootBundle.loadString(path);
  }
}
