import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'dart:async';

import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:yolov4tflite/yolov4tflite.dart';

void main() {
  runApp(
    MaterialApp(home: MyApp()),
  );
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String _platformVersion = 'Unknown';
  bool _busy;
  List<Result> _recogintios;
  File _imagePath;
  double _aspectRatio;
  bool modelLoaded = false;

  @override
  void initState() {
    super.initState();
    initPlatformState();
    loadModel().then((value) => modelLoaded = true);
    _busy = false;
    _recogintios = new List();
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

  Future<void> loadModel() async {
    String labels = await getFileData("assets/labels.txt");
    labels = labels
        .replaceAll(new RegExp("[\n]"), ",")
        .replaceAll(new RegExp("[\t\r]"), "");

    await Yolov4tflite.loadModel(
        modelPath: "assets/yolov4-416-fp32.tflite",
        labels: labels,
        isQuantized: true,
        isTiny: true,
        minimumConfidence: 0.3,
        useNNAPI: false,
        useGPU: false,
        nummberOfThreads: 6);
  }

  Future openImagePicker() async {
    var image = await ImagePicker.pickImage(source: ImageSource.gallery);
    var decodedImage = await decodeImageFromList(image.readAsBytesSync());
    print(decodedImage.width);
    print(decodedImage.height);
    if (image == null) return;
    setState(() {
      _busy = true;
      _imagePath = image;
      _aspectRatio = decodedImage.height / decodedImage.width;
      _recogintios = new List();
    });
    predictImage(image);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Plugin example app'),
      ),
      body: Column(children: [
        _imagePath == null
            ? Container()
            : Stack(
                children: [
                  AnimatedOpacity(
                    child: Image.file(_imagePath),
                    duration: Duration(seconds: 1),
                    opacity: _busy ? 0.7 : 1.0,
                  ),
                  _busy
                      ? Center(child: CircularProgressIndicator())
                      : Container(),
                  Column(
                    children: List<Widget>.generate(
                        _recogintios.length,
                        (index) => new CustomPaint(
                            painter: RectPainter(
                                _recogintios[index].rect,
                                MediaQuery.of(context).size.width,
                                MediaQuery.of(context).size.width *
                                    _aspectRatio,
                                _recogintios[index].name,
                                _recogintios[index].confidence))),
                  )
                ],
                alignment: Alignment.topLeft,
              ),
        _busy || _recogintios.length == 0
            ? Expanded(
                child: Center(
                  child: Text('Running on: $_platformVersion\n'),
                ),
              )
            : Container(),
        _recogintios.length == 0
            ? Container()
            : Expanded(
                child: ListView.builder(
                    itemCount: _recogintios.length,
                    itemBuilder: (context, index) {
                      return ListTile(
                        title: Text(_recogintios[index].name),
                        subtitle: LinearProgressIndicator(
                          value: _recogintios[index].confidence,
                        ),
                      );
                    }),
              )
      ]),
      floatingActionButton: FloatingActionButton(
        onPressed: openImagePicker,
        tooltip: 'Pick Image',
        child: Icon(Icons.image),
      ),
    );
  }

  void predictImage(File image) async {
    String path = image.path;

    // model shuld be loded
    if (!modelLoaded) return;

    var startTime = DateTime.now();

    var r = await Yolov4tflite.detectObjects(imagePath: path);

    print(DateTime.now().difference(startTime));
    print(r);

    List<Result> results = new List();

    r.forEach((element) {
      results.add(Result.fromJson(jsonDecode(element)));
    });

    setState(() {
      _busy = false;
      _recogintios = results;
    });
  }

  Future<String> getFileData(String path) async {
    return await rootBundle.loadString(path);
  }
}

class RectPainter extends CustomPainter {
  Map rect;
  double heigth;
  double width;
  String object;
  double confidence;
  RectPainter(this.rect, this.width, this.heigth, this.object, this.confidence);
  @override
  void paint(Canvas canvas, Size size) {
    if (rect != null) {
      final paint = Paint();
      paint.color = Colors.yellow;
      paint.style = PaintingStyle.stroke;
      paint.strokeWidth = 2.0;
      double l, t, r, b;
      if (this.width > this.heigth) {
        l = rect["l"] / (416 / width);
        b = rect["b"] / (416 / heigth);
        t = rect["t"] / (416 / heigth);
        r = rect["r"] / (416 / width);
      } else {
        // Yolo detector rotates image if width < heigth
        l = width - rect["t"] / (416 / width);
        t = rect["l"] / (416 / heigth);
        r = width - rect["b"] / (416 / width);
        b = rect["r"] / (416 / heigth);
      }

      TextSpan span = TextSpan(
          text: object + " " + (confidence * 100).toString() + "%",
          style: TextStyle(
              color: Colors.black,
              fontSize: 8,
              backgroundColor: Colors.yellow));
      TextPainter tp = TextPainter(
          text: span,
          textAlign: TextAlign.left,
          textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(canvas, Offset(this.width > this.heigth ? l : r, t - 10));

      Rect rect1 = Rect.fromLTRB(l, t, r, b);
      canvas.drawRect(rect1, paint);
    }
  }

  @override
  bool shouldRepaint(RectPainter oldDelegate) => oldDelegate.rect != rect;
}

class Result {
  String name;
  double confidence;
  Map<dynamic, dynamic> rect;

  Result(this.name, this.confidence, this.rect);

  Result.fromJson(Map<dynamic, dynamic> json)
      : name = json["detectedClass"],
        confidence = json["confidenceInClass"],
        rect = json["rect"];
}
