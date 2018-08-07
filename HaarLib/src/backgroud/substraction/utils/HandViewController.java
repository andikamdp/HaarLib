/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the templateasdasdasd in the editor.
 */
package backgroud.substraction.utils;

import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.TrainData;
import org.opencv.videoio.VideoCapture;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class HandViewController implements Initializable {

    @FXML
    private ImageView layarBW;
    @FXML
    private ImageView layarEdge;
    @FXML
    private Button btnStartCamera;
    @FXML
    private Button btnUpdateCamera;
    @FXML
    private TextField txtH;
    @FXML
    private TextField txtV;
    @FXML
    private TextField txtS;
    @FXML
    private TextField txtValue;
    @FXML
    private ImageView layarMain;
//
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private double aWeight;
    private int top, right, bottom, left;
    private int num_frame;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        capture = new VideoCapture();
        cameraActive = false;
        aWeight = 0.5;
        top = 10;
        right = 350;
        bottom = 225;
        left = 590;
        num_frame = 0;

    }

    @FXML
    private void StartCameraOnClick(ActionEvent event) {
        if (!cameraActive) {
            capture.open(0);
            if (this.capture.isOpened()) {
                cameraActive = true;

                Runnable frameGrabber = new Runnable() {
                    @Override
                    public void run() {
                        Mat frame = grabFrame();
                        //# flip the frame so that it is not the mirror view
                        Core.flip(frame, frame, 1);
                        createBox(frame);
                        Mat hand = getBox(frame.clone());

                        Image imageToMat = Utils.mat2Image(frame);

                        updateImageView(layarMain, imageToMat);
                        hand = segment(hand);
                        imageToMat = Utils.mat2Image(hand);
                        updateImageView(layarEdge, imageToMat);

                    }
                };
                this.timer = Executors.newSingleThreadScheduledExecutor();
                timer.scheduleAtFixedRate(frameGrabber, 0, 33,
                        TimeUnit.MILLISECONDS);
                btnStartCamera.setText("stop Camera");

            } else {
                System.err.println("tidak dapat membuka kamera");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.btnStartCamera.setText("Start Camera");

            // stop the timer
            this.stopAcquisition();
        }
    }

    @FXML
    private void UpdateCameraOnClick(ActionEvent event) {
        //mengambil gambar background
        Imgcodecs.imwrite("E:\\TA\\opencv.jpg", grabFrame());
    }
/////////////
//method untuk menggambar kotak untuk posisi tangan

    private Mat createBox(Mat frame) {
        Imgproc.rectangle(frame,
                new Point(frame.cols(), 10), new Point(
                        frame.cols() / 2, frame.rows() - (frame.rows() / 3) - 10),
                new Scalar(0, 0, 255),
                3);
        return frame;
    }
//method untuk mengambil posisi tangan pada kamere

    private Mat getBox(Mat frame) {
        Rect rectCrop = new Rect(new Point(frame.cols() - 5, 10 + 5), new Point(
                frame.cols() / 2 + 5, frame.rows() - (frame.rows() / 3) - 10 - 5)
        );
        frame = frame.submat(rectCrop);
        //# convert the roi to grayscale and blur it
//        Imgproc.accumulateWeighted(frame, frame, 0.5);
        Imgproc.GaussianBlur(frame, frame, new Size(7, 7), 0);
        return frame;
    }
//////

    private Mat cleaning(Mat frame) {
        Mat kernel = new Mat(new Size(3, 3), CvType.CV_16S, new Scalar(255));
        Imgproc.morphologyEx(frame, frame, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(frame, frame, Imgproc.MORPH_OPEN, kernel);
        Imgproc.dilate(frame, frame, kernel);
        return frame;
    }

    private void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println(
                        "Exception in stopping the frame capture, trying to release the camera now... "
                        + e);
            }
        }

        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
        }
    }

    //update tampilan pada frame utama
    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

    /**
     * On application close, stop the acquisition from the camera
     */
    protected void setClosed() {
        this.stopAcquisition();
    }

    private Mat grabFrame() {
        // init everything
        Mat frame = new Mat();
        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BayerBG2BGR);

                }

            } catch (Exception e) {
                // log the error
                System.err.println("Exception during the image elaboration: "
                        + e);
            }
        }
        return frame;
    }

    /**
     * Stop the acquisition from the camera and release all the resources
     */
    public Image bg;
    public Mat diff, diff2, treshold;
//method untuk memisahkan objek dengan background

    private Mat segment(Mat frame) {
        double tres = 50.0;
        Mat frame2 = frame.clone();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        diff = Imgcodecs.imread("E:\\TA\\opencv.jpg");

        Imgproc.cvtColor(diff, diff, Imgproc.COLOR_BGR2GRAY);
        Core.flip(diff, diff, 1);

        Mat dist = new Mat();
        diff = getBox(diff);
        Core.absdiff(diff, frame, dist);
//batas minimum treshold
        layarBW.setImage(Utils.mat2Image(diff));

        Imgproc.threshold(dist, frame, tres, 255, Imgproc.THRESH_BINARY);
        ///
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(frame.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        System.out.println("size contour" + contours.size());

        if (hierarchy.size().height > 0 && hierarchy.size().width > 0) {
            // for each contour, display it in blue
            for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
                Imgproc.
                        drawContours(frame2, contours, idx,
                                new Scalar(250, 0, 0), 3);
            }
        }
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BayerBG2BGR);
        cleaning(frame2);
        HandRec(contours, frame2);

        return frame2;
    }
//method untuk deteksi tangan
//method menghasilhan kooddinatt untuk convexhull

    private void HandRec(List<MatOfPoint> contours, Mat frame) {
        List<MatOfPoint> hullList = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            Point[] contourArray = contour.toArray();
            Point[] hullPoints = new Point[hull.rows()];
            List<Integer> hullContourIdxList = hull.toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            hullList.add(new MatOfPoint(hullPoints));
        }
        Mat drawing = Mat.zeros(frame.size(), CvType.CV_8UC3);
        for (int i = 0; i < contours.size(); i++) {

            Imgproc.drawContours(frame, hullList, i, new Scalar(0, 255, 0), 3);
        }
        System.out.println(hullList.size() + "size hullist");

    }
    int name = 0;

    //method menggambil gambar(image capture)
    @FXML
    private void capturePictureOnSction(ActionEvent event) {

//        Imgcodecs.imwrite("E:\\TA\\" + name + ".jpg", grabFrame());
//        name++;
//        Mat compare = new Mat();
//        Mat result = Imgcodecs.imread("E:\\TA\\15.jpg");
//
//        Imgproc.cvtColor(result, result, Imgproc.COLOR_BGR2GRAY);
//
//        for (int i = 0; i < 15; i++) {
//            compare = Imgcodecs.imread("E:\\TA\\" + i + ".jpg");
//            Imgproc.cvtColor(compare, compare, Imgproc.COLOR_BGR2GRAY);
//            Core.absdiff(result, compare, result);
//        }
//        Imgproc.threshold(result, result, 50.0, 255, Imgproc.THRESH_BINARY);
//        Imgcodecs.imwrite("E:\\TA\\resCom.jpg", result);
//
        Mat bg = Imgcodecs.imread("E:\\TA\\bg.jpg");
        Mat hand = Imgcodecs.imread("E:\\TA\\0.jpg");
        Mat tresholded = hand.clone();
        Mat crop;
        System.out.println(hand.rows() + " " + hand.width());
        System.out.println(hand.cols() + " " + hand.height());
        createBox(hand);
        crop = getBox(hand);
        crop = segment(crop, "E:\\TA\\bg.jpg");
//        hand = segment(tresholded, "E:\\TA\\bg.jpg");
        layarEdge.setImage(Utils.mat2Image(getBox(hand)));
//         tryi(tresholded);
        layarMain.setImage(Utils.mat2Image(hand));
    }

    private Mat segment(Mat frame, String lokasi) {
        double tres = 50.0;
        Mat frame2 = frame.clone();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        diff = Imgcodecs.imread(lokasi);

        Imgproc.cvtColor(diff, diff, Imgproc.COLOR_BGR2GRAY);
        Core.flip(diff, diff, 1);

        Mat dist = new Mat();
        diff = getBox(diff);
        Core.absdiff(diff, frame, dist);
//batas minimum treshold
        layarEdge.setImage(Utils.mat2Image(dist));

        Imgproc.threshold(dist, frame, tres, 255, Imgproc.THRESH_BINARY);
        ///
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(frame.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        System.out.println("size contour" + contours.size());

//        if (hierarchy.size().height > 0 && hierarchy.size().width > 0) {
//            // for each contour, display it in blue
//            for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
//                Imgproc.
//                        drawContours(frame2, contours, idx,
//                                new Scalar(250, 0, 0), 3);
//            }
//        }
        layarBW.setImage(Utils.mat2Image(frame));
//        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BayerBG2BGR);

//        HandRec(contours, frame2);
        return frame;
    }

    public void tryi(Mat frame) {

    }
}
