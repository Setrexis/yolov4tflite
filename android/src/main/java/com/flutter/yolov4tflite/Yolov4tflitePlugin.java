package com.flutter.yolov4tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.renderscript.Type;
import android.util.Log;
import android.app.Activity;


import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.plugin.common.PluginRegistry.Registrar;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.ArrayList;

import com.flutter.yolov4tflite.*;
import com.flutter.yolov4tflite.Classifier.*;

public class Yolov4tflitePlugin implements MethodCallHandler {
    private final Registrar mRegistrar;
    private final Activity activity;
    public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.1f;
    private static Classifier detector;
    private static boolean modalLoaded = false;

    public static void registerWith(Registrar registrar) {
        final MethodChannel channel = new MethodChannel(registrar.messenger(), "yolov4tflite");
        channel.setMethodCallHandler(new Yolov4tflitePlugin(registrar,registrar.activity()));
    }
    
    private Yolov4tflitePlugin(Registrar registrar, Activity activity) {
        this.mRegistrar = registrar;
        this.activity = activity;
    }

    @Override
    public void onMethodCall(MethodCall call, Result result) {
        if (call.method.equals("loadModel")) {
            String modelPath = call.argument("model");
            String labels = call.argument("labels");
            Boolean isTiny = call.argument("isTiny");
            loadModel(modelPath,labels,isTiny,result);
        } else if (call.method.equals("detectObjects")){
            String imgPath = call.argument("image");
            detectobjects(imgPath,result);
        }else if (call.method.equals("getPlatformVersion")) {
            result.success("Android " + android.os.Build.VERSION.RELEASE);
        } else {
            result.notImplemented();
        }
    }

    protected void loadModel(final String path, final String labels,boolean isTiny ,final Result result){
        new Thread(new Runnable(){
            public void run(){
                try {
                    AssetManager assetManager = mRegistrar.context().getAssets();
                    String modalPathKey = mRegistrar.lookupKeyForAsset(path);
                    ByteBuffer modalData = loadFile(assetManager.openFd(modalPathKey));
                    detector = YoloV4Classifier.create(modalData,labels,true);
                    //detector.isTiny = isTiny;
                    modalLoaded=true;
                    activity.runOnUiThread(new Runnable(){
                        public void run(){
                            result.success("Modal Loaded Sucessfully");
                        }
                    });
                } catch (Exception e) {
                    e.printStackTrace();
                    final String msg = e.getMessage();
                    activity.runOnUiThread(new Runnable(){
                        public void run(){
                            result.error("Modal failed to loaded", msg, null);
                        }
                    });
                }
            }
            
        }).start();
    }

    public ByteBuffer loadFile(AssetFileDescriptor fileDescriptor) throws IOException {
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void detectobjects(final String imagePath, final Result result) {
        new Thread(new Runnable(){
            public void run(){
                if (!modalLoaded){
                    result.error("Model is not loaded", "Please load the model before using this methode.", null);
                    return;
                }
                try {
                    //String imagePathKey = mRegistrar.lookupKeyForAsset(imagePath);
                    Bitmap image = Bitmap.createBitmap(BitmapFactory.decodeFile(imagePath));
                    List<Recognition> prediction = detector.recognizeImage(image);
                    System.out.println(prediction);
                    List<String> pred = new ArrayList();
                    for (Recognition r : prediction) {
                        pred.add(r.toString());
                    }
                    final List<String> fpred = pred;
                    activity.runOnUiThread(new Runnable(){
                        public void run(){
                            result.success(fpred);
                        }
                    });
                    
                } catch (Exception e) {
                    e.printStackTrace();
                    final String msg = e.getMessage();
                    activity.runOnUiThread(new Runnable(){
                        public void run(){
                            result.error("Running model failed", msg, null);
                        }
                    });
                }
            }
        }).start();
        
    }
}