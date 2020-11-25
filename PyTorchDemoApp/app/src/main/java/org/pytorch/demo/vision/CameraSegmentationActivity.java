package org.pytorch.demo.vision;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewStub;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.view.ResultRowView;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Map;
import java.util.Queue;

import androidx.annotation.Nullable;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

public class CameraSegmentationActivity extends AbstractCameraXActivity<CameraSegmentationActivity.AnalysisResult> {

    public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
    public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";

    private static final int INPUT_TENSOR_WIDTH = 224;
    private static final int INPUT_TENSOR_HEIGHT = 224;
    private static final int TOP_K = 3;
    private static final int MOVING_AVG_PERIOD = 10;
    private static final String FORMAT_MS = "%dms";
    private static final String FORMAT_AVG_MS = "avg:%.0fms";

    private static final String FORMAT_FPS = "%.1fFPS";
    public static final String SCORES_FORMAT = "%.2f";

    private static final int CLASSNUM = 21;
    private static final int DOG = 12;
    private static final int PERSON = 15;
    private static final int SHEEP = 17;

    static class AnalysisResult {

        private final String[] topNClassNames;
        private final float[] topNScores;
        private final long analysisDuration;
        private final long moduleForwardDuration;

        public AnalysisResult(String[] topNClassNames, float[] topNScores,
                              long moduleForwardDuration, long analysisDuration) {
            this.topNClassNames = topNClassNames;
            this.topNScores = topNScores;
            this.moduleForwardDuration = moduleForwardDuration;
            this.analysisDuration = analysisDuration;
        }
    }

    private boolean mAnalyzeImageErrorState;
    private TextView mFpsText;
    private TextView mMsText;
    private TextView mMsAvgText;
    private Module mModule;
    private String mModuleAssetName;
    private FloatBuffer mInputTensorBuffer;
    private Tensor mInputTensor;
    private long mMovingAvgSum = 0;
    private Queue<Long> mMovingAvgQueue = new LinkedList<>();

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_image_segmentation;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        return ((ViewStub) findViewById(R.id.image_classification_texture_view_stub))
                .inflate()
                .findViewById(R.id.image_classification_texture_view);
    }

    ImageView segmentation;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        segmentation = findViewById(R.id.segmentation);

        mFpsText = findViewById(R.id.image_classification_fps_text);
        mMsText = findViewById(R.id.image_classification_ms_text);
        mMsAvgText = findViewById(R.id.image_classification_ms_avg_text);
    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mMovingAvgSum += result.moduleForwardDuration;
        mMovingAvgQueue.add(result.moduleForwardDuration);
        if (mMovingAvgQueue.size() > MOVING_AVG_PERIOD) {
            mMovingAvgSum -= mMovingAvgQueue.remove();
        }

        mMsText.setText(String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration));
        if (mMsText.getVisibility() != View.VISIBLE) {
            mMsText.setVisibility(View.VISIBLE);
        }
        mFpsText.setText(String.format(Locale.US, FORMAT_FPS, (1000.f / result.analysisDuration)));
        if (mFpsText.getVisibility() != View.VISIBLE) {
            mFpsText.setVisibility(View.VISIBLE);
        }

        if (mMovingAvgQueue.size() == MOVING_AVG_PERIOD) {
            float avgMs = (float) mMovingAvgSum / MOVING_AVG_PERIOD;
            mMsAvgText.setText(String.format(Locale.US, FORMAT_AVG_MS, avgMs));
            if (mMsAvgText.getVisibility() != View.VISIBLE) {
                mMsAvgText.setVisibility(View.VISIBLE);
            }
        }
    }

    protected String getModuleAssetName() {
        if (!TextUtils.isEmpty(mModuleAssetName)) {
            return mModuleAssetName;
        }
        final String moduleAssetNameFromIntent = getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
        mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
                ? moduleAssetNameFromIntent
                : "resnet18.pt";

        return mModuleAssetName;
    }

    @Override
    protected String getInfoViewAdditionalText() {
        return getModuleAssetName();
    }

    private Bitmap mBitmap = null;

    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mAnalyzeImageErrorState) {
            return null;
        }

        try {
            if (mModule == null) {
                final String moduleFileAbsoluteFilePath = new File(
                        Utils.assetFilePath(this, getModuleAssetName())).getAbsolutePath();
                mModule = Module.load(moduleFileAbsoluteFilePath);

                // Allocate image buffer
                mInputTensorBuffer =
                        Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);
                mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3, INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH});
            }

            final long startTime = SystemClock.elapsedRealtime();

            // Put image into float buffer.
            TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
                    image.getImage(), rotationDegrees,
                    INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                    mInputTensorBuffer, 0);

            final long moduleForwardStartTime = SystemClock.elapsedRealtime();
            Map<String, IValue> outTensors = mModule.forward(IValue.from(mInputTensor)).toDictStringKey();
            //final Tensor outputTensor = mModule.forward(IValue.from(mInputTensor)).toTensor();
            final long moduleForwardDuration = SystemClock.elapsedRealtime() - moduleForwardStartTime;

            final Tensor outputTensor = outTensors.get("out").toTensor();
            final float[] scoress = outputTensor.getDataAsFloatArray();
            int width = INPUT_TENSOR_WIDTH;
            int height = INPUT_TENSOR_HEIGHT;
            int[] intValues = new int[width * height];
            for (int j = 0; j < width; j++) {
                for (int k = 0; k < height; k++) {
                    int maxi = 0, maxj = 0, maxk = 0;
                    double maxnum = -Double.MAX_VALUE;
                    for (int i = 0; i < CLASSNUM; i++) {
                        if (scoress[i * (width * height) + j * width + k] > maxnum) {
                            maxnum = scoress[i * (width * height) + j * width + k];
                            maxi = i; maxj = j; maxk = k;
                        }
                    }
                    if (maxi == PERSON)
                        intValues[maxj * width + maxk] = 0xFFFF0000;
                    else if (maxi == DOG)
                        intValues[maxj * width + maxk] = 0xFF00FF00;
                    else if (maxi == SHEEP)
                        intValues[maxj * width + maxk] = 0xFF0000FF;
                    else
                        intValues[maxj * width + maxk] = 0x00000000;
                }
            }

            if (mBitmap == null) {
                mBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            }

            Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, width, height, true);
            Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
            outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
            final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), true);


            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    segmentation.setImageBitmap(transferredBitmap);
                }
            });

            //final float[] scores = outputTensor.getDataAsFloatArray();
            final float[] scores = new float[TOP_K];
            final int[] ixs = Utils.topK(scores, TOP_K);

            final String[] topKClassNames = new String[TOP_K];
            final float[] topKScores = new float[TOP_K];
            for (int i = 0; i < TOP_K; i++) {
                final int ix = ixs[i];
                topKClassNames[i] = Constants.IMAGENET_CLASSES[ix];
                topKScores[i] = scores[ix];
            }
            final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
            return new AnalysisResult(topKClassNames, topKScores, moduleForwardDuration, analysisDuration);
        } catch (Exception e) {
            Log.e(Constants.TAG, "Error during image analysis", e);
            mAnalyzeImageErrorState = true;
            runOnUiThread(() -> {
                if (!isFinishing()) {
                    showErrorDialog(v -> CameraSegmentationActivity.this.finish());
                }
            });
            return null;
        }
    }

    @Override
    protected int getInfoViewCode() {
        return getIntent().getIntExtra(INTENT_INFO_VIEW_TYPE, -1);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mModule != null) {
            mModule.destroy();
        }
    }
}
