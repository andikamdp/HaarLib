/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the templateasdasdasd in the editor.
 */
package src.controller;

import src.Utils;
import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
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
import javafx.scene.input.MouseEvent;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import org.opencv.ml.SVMSGD;
import org.opencv.ml.TrainData;
import org.opencv.videoio.VideoCapture;
import src.utils.Preprocessing;

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
    private MainAppController mainAppController;
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private double aWeight;
    private int top, right, bottom, left;
    private int num_frame;
//
    int name = 0;
    int i = 0;
    SVM s;
    List<Integer> devContourIdxList;
    List<MatOfPoint> contous;
    List<MatOfInt4> devOfInt4s;
    List<MatOfPoint> devOfPoints;
    List<Integer> Puncak = new ArrayList<>();
    List<Integer> Lembah = new ArrayList<>();
    public Boolean startCapture;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        capture = new VideoCapture();
        cameraActive = false;
        startCapture = false;
        i = 0;

    }

    public void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }
/////////////
//Method awal untuk membuka kamera dan memanggil method
/////////////

    @FXML
    private void StartCameraOnClick(ActionEvent event) {

//        s = SVM.load("E:\\TA\\hCoba.xml");
        System.out.println("DELETE");
        if (!cameraActive) {
            capture.open(0);
            if (this.capture.isOpened()) {
                cameraActive = true;

                Runnable frameGrabber = new Runnable() {
                    @Override
                    public void run() {
                        Mat frame = null;
                        try {
                            frame = grabFrame();
//                            DELETE(frame);
                            Start(frame);
                        } catch (Exception e) {
                            System.out.println(e);

                        }

//                        s.predict(frame);
//                        Start(frame);
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

    private void DELETE(Mat frame) {

        //#######################
        Core.flip(frame, frame, 1);
        frame = Preprocessing.drawRect(frame);
        Mat hand = Preprocessing.getBox(frame);
        hand = Preprocessing.getEdge(hand);
        Mat hand_2 = Preprocessing.getBox(frame);
        hand_2 = Preprocessing.getEdge_2(hand_2);
//        //#######################
//        Image imageToMat = Utils.mat2Image(frame);
//        updateImageView(layarMain, imageToMat);
        Image imageToMat = Utils.mat2Image(hand_2);
        updateImageView(layarEdge, imageToMat);
        //#######################
//        SVM s = SVM.load("E:\\TA\\hCoba.xml");
//        SVM.;
        try {

            //#######################
            hand.convertTo(hand, CvType.CV_32FC1);
            System.out.println("");
            System.out.println("AWAL DELETE");
            System.out.println("hand.cols() " + hand.cols());
            System.out.println("hand.rows() " + hand.rows());
            System.out.println("hand.chanels() " + hand.channels());
            System.out.println("hand.type() " + hand.type());
            hand.convertTo(hand, CvType.CV_32FC1);
            Mat trainingDataMat = new Mat(1, 480 * 640, CvType.CV_32FC1);
            System.out.println("trainingDataMat.cols() " + trainingDataMat.cols());
            System.out.println("trainingDataMat.rows() " + trainingDataMat.rows());
            System.out.println("trainingDataMat.chanels() " + trainingDataMat.channels());
            System.out.println("trainingDataMat.type() " + trainingDataMat.type());
            float[] trainingData = new float[hand.cols()];
            for (int j = 0; j < hand.cols(); j++) {
                trainingData[j] = (float) hand.get(0, j)[0];
            }
            trainingDataMat.put(0, 0, trainingData);
            System.out.println("trainingData.length DELETE " + trainingData.length);
//            System.out.println("s.predict(trainingDataMat) DELETE " + s.predict(hand));

//        //#######################
            float p = s.predict(trainingDataMat);
            System.out.println("s.predict(trainingDataMat) DELETE " + p);
            Preprocessing.drawJumlahJari(frame, (int) p);
            imageToMat = Utils.mat2Image(frame);
            updateImageView(layarMain, imageToMat);
        } catch (Exception e) {
            System.out.println("private void DELETE(Mat frame)");
            System.out.println(e);
            System.out.println("AKHIR DELETE");
            System.out.println("");
        }
    }
/////////////
//
/////////////

    private void Start(Mat frame) {
        Puncak = new ArrayList<>();
        Image imageToMat;
        Core.flip(frame, frame, 1);
        Preprocessing.drawRect(frame);
        Mat hand = Preprocessing.getBox(frame.clone());
        Mat tresholded = Preprocessing.segment(hand.clone());
        imageToMat = Utils.mat2Image(tresholded);
        updateImageView(layarBW, imageToMat);
        contous = Preprocessing.getContour(tresholded);
        devOfInt4s = Preprocessing.getDevectIndexPoint(contous);
        Preprocessing.toListMatOfPointDevec(contous, devOfInt4s, devContourIdxList);
        List<Point> pointContourSorted = Preprocessing.toListContour(contous.get(0));
        //ambil titik ekstreme
        double x, x_, y, y_;
        pointContourSorted = Preprocessing.sortPointByX(pointContourSorted);
        x = pointContourSorted.get(0).x;
        x_ = pointContourSorted.get(pointContourSorted.size() - 1).x;
        pointContourSorted = Preprocessing.sortPointByY(pointContourSorted);
        y = pointContourSorted.get(0).y;
        y_ = pointContourSorted.get(pointContourSorted.size() - 1).y;
        Point p = new Point(x - 10, y - 10);
        Point p_ = new Point(x_ + 10, y_);
        //

        Mat handView = Preprocessing.drawRect(hand.clone(), p, p_);
//        hapusTitik(contous, Preprocessing.getBox(frame));
        layarMain.setImage(Utils.mat2Image(frame));
        imageToMat = Utils.mat2Image(tresholded);
        updateImageView(layarBW, imageToMat);
        imageToMat = Utils.mat2Image(handView);
        updateImageView(layarEdge, imageToMat);
        hand = Preprocessing.getBox(hand, p, p_);

        captureImage(hand);

    }
/////////////
//method get image from frame
/////////////

    @FXML
    private void UpdateCameraOnClick(ActionEvent event) {
        //mengambil gambar background
//        Imgcodecs.imwrite("E:\\TA\\opencv.jpg", grabFrame());
//        captureImage();
//        imwrite_DELETE();
        i = 0;
    }

    public void captureImage() {
        Imgcodecs.imwrite("E:\\TA\\HandLearnSVM\\handFullTry1\\" + i
                + ".jpg",
                grabFrame());
        i++;
    }

    public void captureImage(Mat frame) {
        Imgcodecs.imwrite("E:\\TA\\HandLearnSVM\\Try1\\" + txtH.getText() + "\\" + txtH.getText() + "_" + i
                + ".jpg",
                frame);
        i++;
    }

    public void imwrite_DELETE() {
        File folder = new File("E:\\TA\\HandLearnSVM\\leapGestRecog\\train\\" + txtH.getText());
        File[] listOfFiles = folder.listFiles();
/////////////
//prepate Mat
/////////////
        Mat trainingDataMat = new Mat(listOfFiles.length, 10, CvType.CV_32FC1);
        System.out.println(trainingDataMat.type());
        for (int i = 0; i < listOfFiles.length; i++) {
            /////////////
            //
            /////////////
            Mat hand = Imgcodecs.imread("E:\\TA\\HandLearnSVM\\leapGestRecog\\train\\" + txtH.getText() + "\\"
                    + listOfFiles[i].getName());

            System.out.println(hand.type());
            hand = Preprocessing.getEdge_2(hand);
            System.out.println(hand.type());
            Imgcodecs.imwrite("E:\\TA\\HandLearnSVM\\leapGestRecog\\train\\" + txtH.getText() + "\\EDGE\\" + listOfFiles[i].getName() + "_EDGE.jpg", hand);
        }
    }

/////////////
//
/////////////
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

/////////////
//update tampilan pada frame utama
/////////////
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
/////////////
//
/////////////
    private Mat HandRec(List<MatOfPoint> contours, Mat frame) {
        try {

//            List<MatOfInt> hullList = getHullIndexPoint(contours);
//            drawPointColor(toListMatOfPointHull(contours, hullList), frame);
            ///////
            List<MatOfInt4> devList = Preprocessing.getDevectIndexPoint(contours);
            List<MatOfPoint> point = Preprocessing.toListMatOfPointDevec(contours, devList, devContourIdxList);
            Preprocessing.drawPointColor(point, frame);
        } catch (Exception e) {
            System.out.println("HandRec(List<MatOfPoint> contours, Mat frame)");
            System.out.println(e);
            System.out.println("");
        }

        return frame;
    }

/////////////
//method untuk menukar image pada layarBW ke Main
/////////////
    @FXML
    private void BwToMn(MouseEvent event) {
        Image Mn = layarBW.getImage();
        layarBW.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }

/////////////
//method untuk menukar image pada layarEdge ke Main
/////////////
    @FXML
    private void EdgeToMn(MouseEvent event) {
        Image Mn = layarEdge.getImage();
        layarEdge.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }
    ///

    /*
    method untuk mencoba pada gambar
     */
    //method menggambil gambar(image capture)melalui button
    @FXML
    private void capturePictureOnSction(ActionEvent event) {
        Puncak = new ArrayList<>();
        String bg = "E:\\TA\\h0.jpg";
        Mat hand;
        if (txtH.getText().isEmpty()) {
            hand = Imgcodecs.imread("E:\\TA\\HandLearnSVM\\penuh\\hfull0.jpg");
        } else {
            hand = Imgcodecs.imread("E:\\TA\\HandLearnSVM\\penuh\\hfull" + txtH.getText() + ".jpg");
        }

//        hand = Detect_Skin(hand);
//        Imgproc.cvtColor(hand, hand, Imgproc.COLOR_GRAY2BGR);
//        Core.flip(hand, hand, 1);
//        Start(hand);
//        hand = getEdge(hand);
//        Image imageToMat = Utils.mat2Image(hand);
//        updateImageView(layarMain, imageToMat);
//        SVMTry(hand);
//        DELETE(hand);
//        getTrainSVMEdge("01_palm");
//        predict_DELETE();
    }

/////////////
//
/////////////
    public Mat Detect_Skin(Mat hand) {
        //        Mat hand = Imgcodecs.imread("E:\\TA\\HandLearnSVM\\hfull" + txtH.
//        getText() + ".jpg");

        Imgproc.cvtColor(hand, hand, Imgproc.COLOR_BGR2HSV);
        Image imageToMat = Utils.mat2Image(hand);
        updateImageView(layarMain, imageToMat);
//        Core.inRange(hand, new Scalar(Double.valueOf(txtS.getText()),
//                Double.valueOf(txtV.getText()), Double.valueOf(txtValue.getText())),
//                new Scalar(255, 255, 255),
//                hand);
        Core.inRange(hand, new Scalar(0, 21, 50),
                new Scalar(255, 255, 255),
                hand);
        imageToMat = Utils.mat2Image(hand);
        updateImageView(layarBW, imageToMat);
        /////
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
                new Size(11, 11));
//        Imgproc.erode(hand, hand, kernel, new Point(0, 0), 2);
//        Imgproc.dilate(hand, hand, kernel, new Point(0, 0), 2);
//        hand = cleaning(hand);
        imageToMat = Utils.mat2Image(hand);
        updateImageView(layarEdge, imageToMat);
        /////
//        Imgproc.GaussianBlur(hand, hand, new Size(3, 3), 0);
//        Core.bitwise_and(hand_asli, hand, hand_asli);

        imageToMat = Utils.mat2Image(hand);
        updateImageView(layarEdge, imageToMat);
        return hand;
    }

    private void hapusTitik(List<MatOfPoint> contours, Mat hand) {
        try {
            Puncak = new ArrayList<>();
            Point[] point = contous.get(0).toArray();
            Puncak.addAll(devContourIdxList);
            Lembah.addAll(devContourIdxList);
//        System.out.println("isi index puncak awal");
//        for (Integer integer : Puncak) {
//            System.out.println(integer);
//        }
            //jika posisi false berarti cari lembah
            //jika posisi true berarti cari puncak
            Boolean puncak = true;
            for (int j = 0; j < Puncak.size(); j++) {
                int index = Puncak.get(j);
                int indexP = 0;
                if (j + 1 < Puncak.size()) {
                    indexP = Puncak.get(j + 1);
                }

                if (index < point.length && indexP < point.length
                        && point[index].y
                        < hand.rows() - 1
                        && point[indexP].y < hand.rows() - 1) {
                    if (puncak) {
                        //jika menemukan puncak index dicaatat
                        if (Preprocessing.arahTitikY(point[index], point[indexP])) {
                            puncak = false;
                            Lembah.set(j, -1);
                        } //jika titik lebih tinggi index sebelumnya dihapus
                        else {
                            Lembah.set(j, -1);
                            Puncak.set(j, -1);
                            devContourIdxList.set(j, -1);
                        }
                    } else {
                        //jika menemukan lembah index dicaatat
                        if (Preprocessing.arahTitikY(point[indexP], point[index])) {
                            Puncak.set(j, -1);
                            puncak = true;
                        } //jika titik lebih tinggi index sebelumnya dihapus
                        else {
                            Lembah.set(j, -1);
                            Puncak.set(j, -1);
                            devContourIdxList.set(j, -1);
                        }
                    }
                } else {
                    Puncak.set(j, -1);
                    Lembah.set(j, -1);
                    devContourIdxList.set(j, -1);
                }
            }
            Integer rem = -1;
//            Puncak.addAll(Lembah);
            Collections.sort(Puncak);

            for (Integer integer : Puncak) {
                System.out.println(integer);
            }
            System.out.println("");
            System.out.println("contous " + contous.get(0).toArray().length);
            while (Puncak.contains(rem)) {
                Puncak.remove(rem);
            }
            System.out.println("");
            for (Integer integer : Puncak) {
                System.out.println(integer);
            }
//        System.out.println("isi index puncak akhir");
//        for (Integer integer : Puncak) {
//            System.out.println(integer);
//        }
//        while (devContourIdxList.contains(rem)) {
//            devContourIdxList.remove(rem);
//        }
            Mat hand2 = hand.clone();
            Preprocessing.drawPointColor(contous, hand2, Puncak);
            Preprocessing.drawJumlahJari(hand, Puncak.size());
//            layarBW.setImage(Utils.mat2Image(hand2));
//            drawPointColor(contous, hand, devContourIdxList);
            layarEdge.setImage(Utils.mat2Image(hand2));
        } catch (Exception e) {
            System.out.
                    println("hapusTitik(List<MatOfPoint> contours, Mat hand)");
            System.out.println(e);
            System.out.println("");
        }

    }

    @FXML
    private void GetPoint(MouseEvent event) {
        txtV.setText(String.valueOf(event.getX()));
        txtS.setText(String.valueOf(event.getY()));
    }

}
