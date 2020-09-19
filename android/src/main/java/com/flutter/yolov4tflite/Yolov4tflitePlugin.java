package com.flutter.yolov4tflite;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.renderscript.Type;
import android.util.Log;

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

import com.flutter.yolov4tflite.*;
import com.flutter.yolov4tflite.Classifier.*;

public class Yolov4tflitePlugin implements MethodCallHandler {
    private final Registrar mRegistrar;
    public static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static Classifier detector;
    private static boolean modalLoaded = false;

    public static void registerWith(Registrar registrar) {
        final MethodChannel channel = new MethodChannel(registrar.messenger(), "yolov4tflite");
        channel.setMethodCallHandler(new Yolov4tflitePlugin(registrar));
    }
    
    private Yolov4tflitePlugin(Registrar registrar) {
        this.mRegistrar = registrar;
    }

    @Override
    public void onMethodCall(MethodCall call, Result result) {
        if (call.method.equals("loadModel")) {
            String modelPath = call.argument("model");
            String labels = call.argument("labels");
            loadModel(modelPath,labels,result);
        } else if (call.method.equals("detectObjects")){
            String imgPath = call.argument("image");
            detectobjects(imgPath,result);
        }else{
            System.out.println("No method found");
        }
    }

    protected void loadModel(final String path, final String labels, final Result result){
        try {
            AssetManager assetManager = mRegistrar.context().getAssets();
            String modalPathKey = mRegistrar.lookupKeyForAsse(path);
            ByteBuffer modalData = loadFile(assetManager.openFd(modalPathKey));
            detector = YoloV4Classifier.create(modalData,labels,false);
            modalLoaded=true;
            result.success("Modal Loaded Sucessfully");
        } catch (Exception e) {
            e.printStackTrace();
            result.error("Modal failed to loaded", e.getMessage(), null);
        }
    }

    public ByteBuffer loadFile(AssetFileDescriptor fileDescriptor) throws IOException {
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void detectobjects(final String imagePath, final Result result) {
        if (!modalLoaded){
            result.error("Model is not loaded", "Please load the model before using this methode.", null);
            return;
        }
        try {
            String imagePathKey = mRegistrar.lookupKeyForAsset(imagePath);
            Bitmap image = Bitmap.createBitmap(BitmapFactory.decodeFile(imagePathKey));
            List<Recognition> prediction = detector.recognizeImage(image);
            System.out.println(prediction);
            result.success("prediction");
        } catch (Exception e) {
            e.printStackTrace();
            result.error("Running model failed", e.getMessage(), null);
        }
    }
}