library yolov4tflite;

import 'dart:async';
import 'package:meta/meta.dart';
import 'package:flutter/services.dart';

class Yolov4tflite {
  static const MethodChannel _channel = const MethodChannel('yolov4tflite');

  static Future<String> loadModel(
      {@required String model, String labels = ""}) async {
    return await _channel.invokeMethod(
      'loadModel',
      {
        "model": model,
        "labels": labels,
      },
    );
  }

  static Future<String> detectObjects({@required String image}) async {
    return await _channel.invokeMethod('detectObjects', {"image": image});
  }
}
