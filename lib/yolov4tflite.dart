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
      bool isTiny = true}) async {
    return await _channel.invokeMethod(
      'loadModel',
      {
        "model": modelPath,
        "labels": labels,
        "isTiny": isTiny,
      },
    );
  }

  static Future<List<dynamic>> detectObjects(
      {@required String imagePath}) async {
    return await _channel.invokeMethod('detectObjects', {"image": imagePath});
  }
}
