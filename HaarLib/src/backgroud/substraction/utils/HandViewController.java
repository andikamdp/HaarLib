/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the templateasdasdasd in the editor.
 */
package backgroud.substraction.utils;

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
                        Puncak = new ArrayList<>();
                        Image imageToMat;
                        Mat frame = grabFrame();
                        //# flip the frame so that it is not the mirror view
                        Core.flip(frame, frame, 1);
                        createBox(frame);
                        Mat hand = getBox(frame.clone());
                        /////////////////////////////////

                        Mat tresholded = segment(hand.clone());
                        imageToMat = Utils.mat2Image(tresholded);
                        updateImageView(layarBW, imageToMat);
                        //////////////////////////////////////
                        contous = getContour(tresholded);
                        devOfInt4s = getDevectIndexPoint(contous);

                        toListMatOfPointDevec(contous, devOfInt4s);
//                        HandRec(contous, hand);
//                        hitungJarakTitik(devOfPoints);
                        hapusTitik(contous, getBox(frame));
//                        Mat handEdge = HandRec(contous, getBox(hand.clone()));
//        layarBW.setImage(Utils.mat2Image(tresholded));
//                        drawJumlahJari(contous, frame, Puncak);
                        layarMain.setImage(Utils.mat2Image(frame));
                        imageToMat = Utils.mat2Image(hand);
                        updateImageView(layarEdge, imageToMat);
//////// ??

                        //////
                        //////
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

    private Mat segment(Mat frameAsli) {
        double tres = 50.0;
        Mat frameUbah = frameAsli.clone();
        Imgproc.cvtColor(frameAsli, frameAsli, Imgproc.COLOR_BGR2GRAY);

        Mat dist = new Mat();
//        batas minimum treshold
        Imgproc
                .threshold(frameAsli, frameAsli, 100, 255,
                        Imgproc.THRESH_BINARY_INV);

        cleaning(frameAsli);

        return frameAsli;

    }
//method untuk deteksi tangan
//method menghasilhan kooddinatt untuk convexhull

    private Mat HandRec(List<MatOfPoint> contours, Mat frame) {
        try {

//            List<MatOfInt> hullList = getHullIndexPoint(contours);
//            drawPointColor(toListMatOfPointHull(contours, hullList), frame);
            ///////
            List<MatOfInt4> devList = getDevectIndexPoint(contours);
            List<MatOfPoint> point = toListMatOfPointDevec(contours, devList);
            drawPointColor(point, frame);
        } catch (Exception e) {
            System.out.println("HandRec(List<MatOfPoint> contours, Mat frame)");
            System.out.println(e);
            System.out.println("");
        }

        return frame;
    }

    @FXML
    private void BwToMn(MouseEvent event) {
        Image Mn = layarBW.getImage();
        layarBW.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }

    @FXML
    private void EdgeToMn(MouseEvent event) {
        Image Mn = layarEdge.getImage();
        layarEdge.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }
    ///
    int name = 0;
//
//get list point from dev
//28/08/2018

    private List<MatOfPoint> getContour(Mat frame) {
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(frame.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//        System.out.println("size contour" + contours.size());
        //
        //
        MatOfPoint c = contours.get(0);
        Point[] cPoint = c.toArray();

        return contours;
    }
//
//get list point from dev
//28/08/2018

    private List<MatOfInt4> getDevectIndexPoint(List<MatOfPoint> contours
    ) {

        List<MatOfInt4> devList = new ArrayList<>();
        List<MatOfInt> hullList = getHullIndexPoint(contours);
//        for (int i = 0; i < hullList.size(); i++) {
//            try {
//                MatOfInt4 dev = new MatOfInt4();
//                Imgproc.convexityDefects(contours.get(i), hullList.get(i), dev);
////                int[] devarr = dev.toArray();
////                for (int j = 0; j < devarr.length; j++) {
////                    System.out.println(devarr[j]);
////                }
//                devList.add(dev);
//            } catch (Exception e) {
//                System.out.println("isi devec");
//                System.out.println(e);
//            }
//        }
        try {
            MatOfInt4 dev = new MatOfInt4();
            MatOfInt hull = hullList.get(0);
            MatOfPoint cont = contours.get(0);
            Imgproc.convexityDefects(cont, hull, dev);
            devList.add(dev);

        } catch (Exception e) {
            System.out.println(
                    "getDevectIndexPoint(List<MatOfPoint> contours");
            System.err.println(e);
            System.out.println("");
        }
        return devList;
    }
//
//get list point from dev
//28/08/2018

    private List<MatOfInt> getHullIndexPoint(List<MatOfPoint> contours
    ) {
        List<MatOfInt> hullList = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            hullList.add(hull);
        }
        return hullList;
    }
    //method untuk menggambar contour
//
//get list point from hull
//28/08/2018

    public List<MatOfPoint> toListMatOfPointHull(List<MatOfPoint> contours,
            List<MatOfInt> hull) {
        List<MatOfPoint> listPoint = new ArrayList<>();
        for (int j = 0; j < hull.size(); j++) {

            Point[] contourArray = contours.get(j).toArray();
            Point[] hullPoints = new Point[hull.get(j).rows()];
            List<Integer> hullContourIdxList = hull.get(j).toList();
            for (int i = 0; i < hullContourIdxList.size(); i++) {
                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
            }
            listPoint.add(new MatOfPoint(hullPoints));

        }

        return listPoint;
    }

//
//get list point from dev
//28/08/2018
    public List<Integer> devContourIdxList;

    public List<MatOfPoint> toListMatOfPointDevec(List<MatOfPoint> contours,
            List<MatOfInt4> dev) {
        List<MatOfPoint> listPoint = new ArrayList<>();
        try {
            devContourIdxList = dev.get(0).toList();
            Collections.sort(devContourIdxList);
        } catch (Exception e) {
            System.out.println(
                    "toListMatOfPointDevec(List<MatOfPoint> contours, "
                    + "            List<MatOfInt4> dev)");
            System.out.println(e);
            System.out.println("");
        }
//        try {
//            for (int j = 0; j < dev.size(); j++) {
////            System.out.println("iterasi " + j);
//                Point[] contourArray = contours.get(0).toArray();
//                Point[] devPoints = new Point[dev.get(0).rows() * 4];
//                devContourIdxList = dev.get(0).toList();
//                Collections.sort(devContourIdxList);
//
////                dev.get(0);
//                for (int i = 0; i < devContourIdxList.size(); i++) {
//                    if (devContourIdxList.get(i) < contourArray.length /*&& (i == 0 || devContourIdxList.get(i)
//                        - devContourIdxList.get(i
//                                - 1) > 5)*/) {
//                        devPoints[i] = contourArray[devContourIdxList.get(i)];
////                    System.out.println("point " + devPoints[i].toString());
//                    } else {
//                        devPoints[i] = new Point(-1, -1);
//                    }
//                }
//                listPoint.add(new MatOfPoint(devPoints));
//            }
//        } catch (Exception e) {
//            System.out.println(
//                    "toListMatOfPointDevec(List<MatOfPoint> contours, List<MatOfInt4> dev)");
//            System.out.println(e);
//            System.out.println("");
////
//            //error mungkint terjadi pada method ini
////        aa //            //
//        }

        return listPoint;
    }

    public void drawContour(List<MatOfPoint> contours, Mat frame
    ) {
        for (int i = 0; i < contours.size(); i++) {

            Imgproc.drawContours(frame, contours, i, new Scalar(0, 255, 0), 3);
        }
    }

    public void drawContour(List<MatOfPoint> contours, Mat frame,
            Integer[] Index
    ) {
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(frame, contours, i, new Scalar(0, 255, 0), 3);
        }
    }
    int i = 0;

    public void captureImage() {
        Imgcodecs.imwrite("E:\\TA\\h" + i + ".jpg", grabFrame());
        i++;
    }
//
//get list point from dev
//28/08/2018

    public void drawPointColor(List<MatOfPoint> contours, Mat frame,
            Integer[] index) {
        for (int i = 0; i < contours.get(0).toArray().length; i++) {
            Scalar s;
            if (i == 0) {
                s = new Scalar(255, 255, 255);
            } else if (i % 3 == 0) {
                s = new Scalar(255, 0, 0);
            } else if (i % 3 == 2) {
                s = new Scalar(0, 255, 0);
            } else {
                s = new Scalar(0, 0, 255);
            }

            if (index[i] != null && contours.get(0).toArray()[i].x >= 0
                    && index[i] >= 0) {
                Imgproc.circle(frame, contours.get(0).toArray()[i], 10,
                        s, -1);
//                Imgproc.putText(frame, contours.get(0).toArray()[i].toString(),
//                        contours.get(0).toArray()[i], 2, 0.5, s);
                System.out.println(index[i]);
            }
        }
    }
//
//get list point from dev
//28/08/2018

    public void drawPointColor(List<MatOfPoint> contours, Mat frame
    ) {
        Scalar s;
        for (int i = 0; i < contours.get(0).toArray().length; i++) {

            if (i == 0) {
                s = new Scalar(255, 255, 255);
            } else if (i % 3 == 0) {
                s = new Scalar(255, 0, 0);
            } else if (i % 3 == 2) {
                s = new Scalar(0, 255, 0);
            } else {
                s = new Scalar(0, 0, 255);
            }
            if (contours.get(0).toArray()[i].x >= 0) {
                Imgproc.circle(frame, contours.get(0).toArray()[i], 10,
                        s, -1);
//                Imgproc.putText(frame, contours.get(0).toArray()[i].toString(),
//                        contours.get(0).toArray()[i], 2, 0.5, s);

            }
        }
    }

    //
//get list point from dev
//28/08/2018
    public void drawPointColor(List<MatOfPoint> contours, Mat frame,
            List<Integer> index) {
//        for (Integer integer : index) {
        Integer k = -1;
        for (int j = 0; j < index.size(); j++) {

            Scalar s;
            if (i == 0) {
                s = new Scalar(255, 255, 255);
            } else if (i % 3 == 0) {
                s = new Scalar(255, 0, 0);
            } else if (i % 3 == 2) {
                s = new Scalar(0, 255, 0);
            } else {
                s = new Scalar(0, 0, 255);
            }
            if (index.get(j) < contous.get(0).toArray().length
                    && index.get(j) >= 0) {
                Imgproc.
                        circle(frame, contous.get(0).toArray()[index.get(j)], 10,
                                s, -1);
            } else {
                k = j;

            }
//                Imgproc.putText(frame, contours.get(0).toArray()[i].toString(),
//                        contours.get(0).toArray()[i], 2, 0.5, s);

        }
        Puncak.remove(k);
    }
//
//get list point from dev
//28/08/2018

    public void drawJumlahJari(List<MatOfPoint> contours, Mat frame,
            List<Integer> index) {
        Scalar s;

        s = new Scalar(0, 0, 255);

        Imgproc.putText(frame, "Jumlah Puncak Jari = " + index.size(),
                new Point(10.0, 10.0), 2, 0.5, s);

//                I
    }
    /*
    method untuk mencoba pada gambar
     */
    //method menggambil gambar(image capture)melalui button
    List<MatOfPoint> contous;
    List<MatOfInt4> devOfInt4s;
    List<MatOfPoint> devOfPoints;

    @FXML
    private void capturePictureOnSction(ActionEvent event) {
        Puncak = new ArrayList<>();
        String bg = "E:\\TA\\h0.jpg";
        Mat hand;
        if (txtH.getText().isEmpty()) {
            hand = Imgcodecs.imread("E:\\TA\\h1.jpg");
        } else {
            hand = Imgcodecs.imread("E:\\TA\\h" + txtH.getText() + ".jpg");
        }
        Core.flip(hand, hand, 1);
        Mat tresholded = hand.clone();
        createBox(tresholded);
        tresholded = getBox(tresholded);
        tresholded = segment(tresholded, bg);
        layarMain.setImage(Utils.mat2Image(hand));
        layarBW.setImage(Utils.mat2Image(tresholded));
        //////
        //////
        contous = getContour(tresholded);
        devOfInt4s = getDevectIndexPoint(contous);
        devOfPoints
                = toListMatOfPointDevec(contous, devOfInt4s);
//        hitungJarakTitik(devOfPoints);
        hapusTitik(contous, getBox(hand.clone()));
        Mat handEdge = HandRec(contous, getBox(hand.clone()));
//        layarBW.setImage(Utils.mat2Image(tresholded));
        drawJumlahJari(contous, hand, Puncak);
        layarMain.setImage(Utils.mat2Image(hand));
//        layarEdge.setImage(Utils.mat2Image(handEdge));
    }

    private Mat segment(Mat frameAsli, String lokasi) {
        double tres = 50.0;
        Mat frameUbah = frameAsli.clone();
        diff = Imgcodecs.imread(lokasi);
        Imgproc.cvtColor(frameAsli, frameAsli, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(diff, diff, Imgproc.COLOR_BGR2GRAY);
        Core.flip(diff, diff, 1);
        Mat dist = new Mat();
        diff = getBox(diff);
//        Core.absdiff(diff, frameAsli, dist);
//        batas minimum treshold
        Imgproc.threshold(frameAsli, frameAsli, 100, 255,
                Imgproc.THRESH_BINARY_INV);

        cleaning(frameAsli);

        return frameAsli;
    }

    private void hitungJarakTitik(List<MatOfPoint> contour) {
        Point[] point = contour.get(0).toArray();
        for (int j = 0; j < point.length - 1; j++) {
            double jarak = Math.sqrt(Math.pow(point[j].x - point[j + 1].x, 2)
                    + Math.
                            pow(
                                    point[j].y
                                    - point[j + 1].y, 2));
        }
    }

    private void hitungJarakTitik(List<MatOfPoint> contour, Integer[] index) {
        Point[] point = contour.get(0).toArray();
        int k = 0;
        for (int j = 1; j < point.length - 1; j++) {
            if (index[j] != null && index[j] < 0) {
                double jarak = Math.sqrt(Math.
                        pow(point[k].x - point[j].x, 2)
                        + Math.pow(point[k].y - point[j].y, 2));
                k = j;
            }
        }
    }

    private double hitungJarakTitik(Point titikA, Point titikB) {

        double jarak = Math.sqrt(Math.pow(titikA.x - titikB.x, 2)
                + Math.pow(titikA.y - titikB.y, 2));

        return jarak;

    }
    List<Integer> Puncak = new ArrayList<>();

    private void hapusTitik(List<MatOfPoint> contours, Mat hand) {
        try {
            Point[] point = contous.get(0).toArray();
            Puncak.addAll(devContourIdxList);
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
                System.out.println("index " + index);
                System.out.println("indexP " + indexP);
                System.out.println("Pointlength " + point.length);
                if (index < point.length && indexP < point.length
                        && point[index].y
                        < hand.rows() - 1
                        && point[indexP].y < hand.rows() - 1) {
                    if (puncak) {
                        //jika menemukan lembah index dicaatat
                        if (arahTitikY(point[index], point[indexP])) {
                            puncak = false;
                        } //jika titik lebih tinggi index sebelumnya dihapus
                        else {
                            Puncak.set(j, -1);
                            devContourIdxList.set(j, -1);
                        }
                    } else {
                        //jika menemukan lembah index dicaatat
                        if (arahTitikY(point[indexP], point[index])) {
                            Puncak.set(j, -1);
                            puncak = true;
                        } //jika titik lebih tinggi index sebelumnya dihapus
                        else {
                            Puncak.set(j, -1);
                            devContourIdxList.set(j, -1);
                        }
                    }
                } else {
                    Puncak.set(j, -1);
                    devContourIdxList.set(j, -1);
                }
            }
            Integer rem = -1;
            Puncak.remove(rem);
            while (Puncak.contains(rem)) {
                Puncak.remove(rem);
            }
//        System.out.println("isi index puncak akhir");
//        for (Integer integer : Puncak) {
//            System.out.println(integer);
//        }
//        while (devContourIdxList.contains(rem)) {
//            devContourIdxList.remove(rem);
//        }
            Mat hand2 = hand.clone();
            drawPointColor(contous, hand, Puncak);
            drawJumlahJari(contous, hand, Puncak);
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
//
//periksa arah titikX
//if (titikA.x <= titikB.x)
//    jika titik pertama lebih kekanan dari titik kedua
//31/08/2018

    public Boolean arauTitikX(Point titikA, Point titikB) {
        if (titikA.x > titikB.x) {
            return true;
        } else {
            return false;
        }
    }
//
//periksa arah titikY
//if (titikA.x <= titikB.x)
//    jika titik pertama lebih keatas dari titik kedua
//31/08/2018

    public Boolean arahTitikY(Point titikA, Point titikB) {
        if (titikA.y < titikB.y) {
//            System.out.print(" Puncak->menurun");
//            System.out.println(titikA.toString() + " " + titikB.toString());
            return true;
        } else if (titikA.y == titikB.y) {
//            System.out.print(" sama");
            return false;
        } else {

//            System.out.println(titikA.toString() + " " + titikB.toString());
//            System.out.print(" Lembah->menaik");
            return false;
        }
    }

}
