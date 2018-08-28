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
                        Image imageToMat;
                        Mat frame = grabFrame();
                        //# flip the frame so that it is not the mirror view
                        Core.flip(frame, frame, 1);
                        createBox(frame);
                        Mat hand = getBox(frame.clone());

                        imageToMat = Utils.mat2Image(frame);

                        updateImageView(layarMain, imageToMat);
                        Mat handTreshold = segment(hand.clone());

                        imageToMat = Utils.mat2Image(handTreshold);
                        updateImageView(layarBW, imageToMat);

                        List<MatOfPoint> contours = getContour(handTreshold);
                        HandRec(contours, hand);
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

    private Mat segment(Mat frameAsli) {
        double tres = 50.0;
        Mat frameUbah = frameAsli.clone();
        diff = Imgcodecs.imread("E:\\TA\\opencv.jpg");
        Imgproc.cvtColor(frameAsli, frameAsli, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(diff, diff, Imgproc.COLOR_BGR2GRAY);
        Core.flip(diff, diff, 1);
        Mat dist = new Mat();
        diff = getBox(diff);

        Core.absdiff(diff, frameAsli, dist);
//        batas minimum treshold
        Imgproc
                .threshold(dist, frameAsli, tres, 255, Imgproc.THRESH_BINARY);

        cleaning(frameAsli);

        return frameAsli;
    }
//method untuk deteksi tangan
//method menghasilhan kooddinatt untuk convexhull

    private Mat HandRec(List<MatOfPoint> contours, Mat frame) {
        try {
            System.out.println("");
            System.out.println("con length " + contours.size());
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
        System.out.println("size contour" + contours.size());
        //
        //
        MatOfPoint c = contours.get(0);
        Point[] cPoint = c.toArray();
        System.out.println("cpoint length" + cPoint.length + "");
        for (int j = 0; j < cPoint.length; j++) {
            System.out.println(cPoint[j].toString());
        }
        return contours;
    }
//
//get list point from dev
//28/08/2018

    private List<MatOfInt4> getDevectIndexPoint(List<MatOfPoint> contours
    ) {

        List<MatOfInt4> devList = new ArrayList<>();
        List<MatOfInt> hullList = getHullIndexPoint(contours);

        System.out.println("contous size " + contours.size());
        System.out.println("hull size " + hullList.size());
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
            System.out.println("papap");
            System.out.println("papap");

        } catch (Exception e) {
            System.out.println(
                    "getDevectIndexPoint(List<MatOfPoint> contours");
            System.out.println(e);
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

    //method menggambil gambar(image capture)melalui button
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
//////
//////
//////
//////
//        Mat bg = Imgcodec
        Mat bg = Imgcodecs.imread("E:\\TA\\bg.jpg");
        Mat hand = Imgcodecs.imread("E:\\TA\\Untitled.jpg");
        Mat tresholded = hand.clone();
        createBox(tresholded);
        tresholded = getBox(tresholded);
        tresholded = segment(tresholded, "E:\\TA\\Untitled - Copy.jpg");
        layarMain.setImage(Utils.mat2Image(hand));
        //////
        //////
        List<MatOfPoint> contous = getContour(tresholded);
        hand = HandRec(contous, getBox(hand));
        layarBW.setImage(Utils.mat2Image(tresholded));
        layarEdge.setImage(Utils.mat2Image(hand));
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

        Core.absdiff(diff, frameAsli, dist);
//        batas minimum treshold
        Imgproc
                .threshold(dist, frameAsli, tres, 255, Imgproc.THRESH_BINARY);

        cleaning(frameAsli);

        return frameAsli;
    }

    private void LacakDevect(List<MatOfPoint> contours, Mat frame) {
        Mat frame2 = frame.clone();
//        layarEdge.setImage(Utils.mat2Image(frame2));
//        List<MatOfPoint> hullList = new ArrayList<>();
//        List<MatOfPoint> devxList = new ArrayList<>();
//        for (MatOfPoint contour : contours) {
//            MatOfInt hull = new MatOfInt();
//            MatOfInt4 devx = new MatOfInt4();
//            //
//            Imgproc.convexHull(contour, hull);
//            Imgproc.convexityDefects(contour, hull, devx);
//            //
//            Point[] contourArray = contour.toArray();
//            Point[] hullPoints = new Point[hull.rows()];
//
//            //
//            List<Integer> hullContourIdxList = hull.toList();
//            List<Integer> devxContourIdxList = devx.toList();
//            Point[] devexPoints = new Point[devxContourIdxList.size()];
//            System.out.println(hullPoints.length + "panjang hull point");
//            System.out.
//                    println(hullContourIdxList.size()
//                            + "panjang hull list point");
//            for (int i = 0; i < hullContourIdxList.size(); i++) {
//                hullPoints[i] = contourArray[hullContourIdxList.get(i)];
//            }
//            //
//            System.out.println(contourArray.length
//                    + "panjang contourArray point");
//            System.out.println(devexPoints.length + "panjang devex point");
//            System.out.
//                    println(devxContourIdxList.size()
//                            + "panjang devex list point");
//            for (int i = 0; i < 10; i++) {
//                devexPoints[i] = contourArray[devxContourIdxList.get(i)];
//            }
//            hullList.add(new MatOfPoint(hullPoints));
//            devxList.add(new MatOfPoint(devexPoints));
//        }
//        Mat drawing = Mat.zeros(frame.size(), CvType.CV_8UC3);
//        for (int i = 0; i < contours.size(); i++) {
//
//            Imgproc.drawContours(frame, hullList, i, new Scalar(0, 255, 0), 3);
////            Imgproc.drawContours(frame, devxList, i, new Scalar(0, 0, 255), 3);
//        }
//        System.out.println(hullList.size() + "size hullist");
//        System.out.println(contours.size() + "size contours");

///////////////
        Image g;

        List<MatOfInt> hullList = new ArrayList<>();
        for (MatOfPoint contour : contours) {
            MatOfInt hull = new MatOfInt();
            Imgproc.convexHull(contour, hull);
            hullList.add(hull);
        }
        //
        ArrayList<MatOfInt4> devList = new ArrayList<>();
        for (int i = 0; i < hullList.size(); i++) {
            MatOfInt4 dev = new MatOfInt4();
            Imgproc.convexityDefects(contours.get(i), hullList.get(i), dev);
            devList.add(dev);
        }
        //print isi list hull untuk convexhull
        System.out.println("");
        System.out.println("hull");
        System.out.println("hull length " + hullList.size());
        for (MatOfInt hull : hullList) {
            System.out.println("row" + hull.rows());
            System.out.println("col" + hull.cols());
            System.out.println("H" + hull.height());
            System.out.println("W" + hull.width());
        }
        //print isi list devList untuk devec convexity
        System.out.println("");
        System.out.println("dev");
        System.out.println("dev length " + devList.size());
        for (MatOfInt4 dev : devList) {
            System.out.println("row" + dev.rows());
            System.out.println("col" + dev.cols());
            System.out.println("H" + dev.height());
            System.out.println("W" + dev.width());
        }
//

        System.out.println("");
        System.out.println("con");
        System.out.println("con length " + contours.size());
        List<MatOfPoint> devListPoint = new ArrayList<>();
        for (int i = 0; i < contours.size(); i++) {
            MatOfInt4 dev = devList.get(i);
            MatOfInt hul = hullList.get(i);
//              Imgproc.convexHull(contour, hull);//
            Point[] contourArray = contours.get(i).toArray();
            Point[] hullPoints = new Point[hul.rows()];
            Point[] devPoints = new Point[dev.rows()];
            //
            //
            System.out.println("");
            System.out.println("contourArray");
            System.out.println("lengtg " + contourArray.length);
            for (int j = 0; j < contourArray.length; j++) {
//                devPoints[j] = contourArray[devContourIdxList.get(j)];
                System.out.println(contourArray[j]);
            }
            //
            //
            List<Integer> devContourIdxList = dev.toList();
            System.out.println("");
            System.out.println("devPoints");
            for (int j = 0; j < devPoints.length; j++) {
//                devPoints[j] = contourArray[devContourIdxList.get(j)];
                System.out.println(devPoints[j]);
            }
            //
            //
            List<Integer> hullContourIdxList = hul.toList();
            System.out.println("");
            System.out.println("hullPoints");
            for (int j = 0; j < hullPoints.length; j++) {
//                devPoints[j] = contourArray[devContourIdxList.get(j)];
                System.out.println(hullPoints[j]);
            }
            //
            //
            System.out.println("");
            System.out.println("devContourIdxList");
            System.out.println("lengtg " + devContourIdxList.size());
            for (int j = 0; j < devContourIdxList.size(); j++) {
//                devPoints[j] = contourArray[devContourIdxList.get(j)];
                System.out.println(devContourIdxList.get(j));
            }
            //
            //
            System.out.println("");
            System.out.println("hullContourIdxList");
            System.out.println("lengtg " + hullContourIdxList.size());
            for (int j = 0; j < hullContourIdxList.size(); j++) {
//                devPoints[j] = contourArray[devContourIdxList.get(j)];
                System.out.println(hullContourIdxList.get(j));
            }
            //
            //
            System.out.println("");
            System.out.println("devContourIdxList");
            System.out.println("size " + devContourIdxList.size());
            for (int j = 0; j < devContourIdxList.size(); j++) {
                if (devContourIdxList.get(j) < 164) {
                    System.out.println(contourArray[devContourIdxList.
                            get(j)]);
                    Imgproc.drawMarker(frame, contourArray[devContourIdxList.
                            get(j)],
                            new Scalar(255, 255, 0));
                }
//                System.out.println(devContourIdxList.get(j));
            }
            //
            //
            System.out.println("");
            System.out.println("hullContourIdxList");
            System.out.println("size " + hullContourIdxList.size());
            for (int j = 0; j < hullContourIdxList.size(); j++) {
                if (hullContourIdxList.get(j) < 164) {
                    System.out.println(contourArray[hullContourIdxList.
                            get(j)]);
                    Imgproc.drawMarker(frame2, contourArray[hullContourIdxList.
                            get(j)],
                            new Scalar(255, 0, 0));
                }
//                System.out.println(devContourIdxList.get(j));
            }
            layarBW.setImage(Utils.mat2Image(frame2));
//            devListPoint.add(new MatOfPoint(devPoints));
            //
            //
        }
    }

    public void convexity_defects(List<MatOfPoint> contours, Mat frame) {
//        Contour contour = contours.get(0);
//        MatOfPoint contour = contours.get(0);
//        convexHull = contour.getPolygonApproximation().getConvexHull();

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

//        hullCenter = new PVector(convexHull.getBoundingBox().x + convexHull.
//                getBoundingBox().width / 2,
//                convexHull.getBoundingBox().y
//                + convexHull.getBoundingBox().height / 2);
//        Rect boundingRect = Imgproc.boundingRect(hullList.get(0));
        PVector hullCenter = new PVector(Imgproc.boundingRect(hullList.get(0)).x
                + Imgproc.boundingRect(hullList.get(0)).width / 2, Imgproc.
                boundingRect(hullList.get(0)).y
                + Imgproc.boundingRect(hullList.get(0)).height / 2);
        MatOfInt hull = new MatOfInt();
        MatOfPoint points = new MatOfPoint(contours.get(0));
        Imgproc.convexHull(points, hull);

        MatOfInt4 defects = new MatOfInt4();
        Imgproc.convexityDefects(points, hull, defects);

        ArrayList<PVector> defectPoints = new ArrayList<PVector>();
        ArrayList<Float> depths = new ArrayList<Float>();

        ArrayList<Integer> defectIndices = new ArrayList<Integer>();

        for (int i = 0; i < defects.height(); i++) {

            int startIndex = (int) defects.get(i, 0)[0];
            int endIndex = (int) defects.get(i, 0)[1];
            int defectIndex = (int) defects.get(i, 0)[2];
            if (defects.get(i, 0)[3] > 10000) {
                defectIndices.add(defectIndex);
                defectPoints.add(new PVector(
                        (float) contours.get(0).toArray()[defectIndex].x,
                        (float) contours.
                                get(0).toArray()[defectIndex].y)
                );
                depths.add((float) defects.get(i, 0)[3]);
            }
        }

        Integer[] handIndices
                = new Integer[defectIndices.size() + hull.height()];
        for (int i = 0; i < hull.height(); i++) {
            handIndices[i] = (int) hull.get(i, 0)[0];
        }
        for (int d = 0; d < defectIndices.size(); d++) {
            handIndices[d + hull.height()] = defectIndices.get(d);
        }
//        drawContour(contours, frame, handIndices);
        int d = 0;

//        Arrays.sort(handIndices);
        ArrayList<PVector> handPoints = new ArrayList<PVector>();
        for (int i = 0; i < handIndices.length; i++) {
            PVector point = new PVector(
                    (float) contours.get(0).toArray()[handIndices[i]].x,
                    (float) contours.get(0).toArray()[handIndices[i]].y);
            handPoints.add(point);
        }
        Point[] p = new Point[handPoints.size()];
        System.out.println("drawContour");
        System.out.println("handPoints.size()" + handPoints.size());
        for (int i = 0; i < handPoints.size(); i++) {
            p[i] = new Point(handPoints.get(i).x, handPoints.get(i).y);
            System.out.println(p[i].x);
            System.out.println(p[i].y);
            System.out.println("pp");
        }
        MatOfPoint po = new MatOfPoint(p);
        List<MatOfPoint> cont = new ArrayList<MatOfPoint>();
        cont.add(po);
        drawContour(cont, frame, handIndices);
    }

    //method untuk menggambar contour
//
//get list point from hull
//28/08/2018
    public List<MatOfPoint> toListMatOfPointHull(List<MatOfPoint> contours,
            List<MatOfInt> hull) {
        List<MatOfPoint> listPoint = new ArrayList<>();
        System.out.println("isi dari dev  " + hull.size());
        System.out.println("isi dari dev row " + hull.get(0).rows());
        System.out.println("isi dari dev row " + hull.get(0).cols());
        System.out.println("isi dari dev row " + hull.get(0).height());
        System.out.println("isi dari dev row " + hull.get(0).width());
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
    public List<MatOfPoint> toListMatOfPointDevec(List<MatOfPoint> contours,
            List<MatOfInt4> dev) {
        List<MatOfPoint> listPoint = new ArrayList<>();
        System.out.println("isi dari dev  " + dev.size());
        System.out.println("isi dari dev row " + dev.get(0).rows());
        System.out.println("isi dari dev row " + dev.get(0).cols());
        System.out.println("isi dari dev row " + dev.get(0).height());
        System.out.println("isi dari dev row " + dev.get(0).width());
////        try {
        for (int j = 0; j < dev.size(); j++) {
//            System.out.println("iterasi " + j);
            Point[] contourArray = contours.get(0).toArray();
            Point[] devPoints = new Point[dev.get(0).rows() * 4];
            List<Integer> devContourIdxList = dev.get(0).toList();
            Collections.sort(devContourIdxList);
            System.out.println("devContourIdxList.size "
                    + devContourIdxList.
                            size());
            System.out.println(devContourIdxList.toString());
            System.out.println("");
            System.out.println("contourArray.length " + contourArray.length);
            System.out.println("devContourIdxList.size() "
                    + devContourIdxList.size());
//                dev.get(0);
            for (int i = 0; i < devContourIdxList.size(); i++) {
                if (devContourIdxList.get(i) < contourArray.length /*&& (i == 0 || devContourIdxList.get(i)
                        - devContourIdxList.get(i
                                - 1) > 5)*/) {
                    devPoints[i] = contourArray[devContourIdxList.get(i)];
//                    System.out.println("point " + devPoints[i].toString());
                } else {
                    devPoints[i] = new Point(-1, -1);
                }
            }
            for (int i = 0; i < devPoints.length; i++) {

                System.out.println("point " + devPoints[i].toString() + "   "
                        + i);

            }
            listPoint.add(new MatOfPoint(devPoints));
        }
//        } catch (Exception e) {
//            System.out.println(e);
//            System.out.println(
//                    "toListMatOfPointDevec(List<MatOfPoint> contours,\n"
//                    + "            List<MatOfInt4> dev)");
//            //
//            //error mungkint terjadi pada method ini
//            //
//
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
            if (contours.get(0).toArray()[i].x >= 0) {
                Imgproc.circle(frame, contours.get(0).toArray()[i], right,
                        new Scalar(0, 255,
                                0), -1);
            }
        }
    }
//
//get list point from dev
//28/08/2018

    public void drawPointColor(List<MatOfPoint> contours, Mat frame
    ) {
        for (int i = 0; i < contours.get(0).toArray().length; i++) {
            if (contours.get(0).toArray()[i].x >= 0) {
                Imgproc.circle(frame, contours.get(0).toArray()[i], 10,
                        new Scalar(0, 0,
                                255), -1);
            }
        }
    }
}
