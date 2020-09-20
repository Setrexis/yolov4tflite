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

  /// Loads the model from [modelPath] and sets tflite options.
  /// [labels] contains the "," separated labels for the model.
  /// Use [isTiny]=false if your model is not tiny.
  /// For Quantized models set [isQuantized]=true.
  /// [inputSize] specify the models input size.
  ///
  /// Use [numberOfThreads] if you want to change models performence.
  /// If you want to use NNAPI set [useNNAPI]=true.
  /// If you want to use GPU set [useGPU]=true.
  /// If you set [useNNAPI] and [useGPU] on true NNAPI is used if avaleble.
  /// For more information about enchancing you models performence visit https://www.tensorflow.org/lite/performance/best_practices.
  ///
  /// If a result of your model is under [minimumConfidence] it wont be returnd.
  static Future<String> loadModel(
      {@required String modelPath,
      String labels = "",
      bool isTiny = true,
      bool isQuantized = false,
      int inputSize = 416,
      double imageMean = 0,
      double imageStd = 255,
      double minimumConfidence = 0.4,
      bool useGPU = false,
      bool useNNAPI = false,
      int nummberOfThreads = 4}) async {
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
        "useNNAPI": useNNAPI,
        "nummberOfThreads": nummberOfThreads,
      },
    );
  }

  /// Runs object detection on model.
  /// [imagePath] provides the path to the image to be analyzed.
  static Future<List<dynamic>> detectObjects(
      {@required String imagePath}) async {
    return await _channel.invokeMethod('detectObjects', {"image": imagePath});
  }

  static Future<String> closeModel() async {
    return await _channel.invokeMethod("close");
  }
}
