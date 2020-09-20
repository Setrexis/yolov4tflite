library yolov4tflite;

import 'dart:async';
import 'package:meta/meta.dart';
import 'package:flutter/services.dart';

class Yolov4tflite {
  static const MethodChannel _channel = const MethodChannel('yolov4tflite');

  static Future<String> get platformVersion async {
    final String version = await _channel.invokeMethod('getPlatformVersion');
    return version;
  }

  static Future<String> loadModel(
      {@required String modelPath,
      String labels = "",
      bool isTiny = true,
      bool isQuantized = false,
      int inputSize = 416,
      double imageMean = 0,
      double imageStd = 255,
      double minimumConfidence = 0.4,
      bool useGPU = false}) async {
    return await _channel.invokeMethod(
      'loadModel',
      {
        "model": modelPath,
        "labels": labels,
        "isTiny": isTiny,
        "isQuantized": isQuantized,
        "inputSize": inputSize,
        "imageMean": imageMean,
        "imageStd": imageStd,
        "minimumConfidence": minimumConfidence,
        "useGPU": useGPU,
      },
    );
  }

  static Future<List<dynamic>> detectObjects(
      {@required String imagePath}) async {
    return await _channel.invokeMethod('detectObjects', {"image": imagePath});
  }
}
