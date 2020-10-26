import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Robot;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.Timer;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.features2d.BFMatcher;
import org.opencv.features2d.ORB;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class OpenCVTester {
	static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }
	static final int FRAME_TIME = 1000 / 30;
	
	private static Mat mat = new Mat();
	private static VideoCapture capture;
	private static BufferedImage img = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
	private static BufferedImage imgSpare = null;
//	private static ArrayList<RectLifespan> boundingBoxes = new ArrayList<RectLifespan>();
	static HashMap<Mat, RectLifespan> keyPointDescriptorList = new HashMap<Mat, RectLifespan>();
	
	private static Robot rob;
	static { try { rob = new Robot(); } catch (Exception e) {} }

	private static JFrame frame = new JFrame();
	private static JPanel panel;
	
	
	static final BufferedImage IMG_NONE = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
	public static void main(String[] args) {
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setExtendedState(JFrame.MAXIMIZED_BOTH);
		panel = new JPanel() {
			private static final long serialVersionUID = 1L;

			final int frameBatch = 10;
			long lastFrame = 0;
			int frameCounter = 0;
			long avgFrameTime = 0;

			public void paint(Graphics g) {
				super.paint(g);
				g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 12));
				
				final int pad = g.getFont().getSize();
				final int sz = 150;
				int locationX = 0;
				int locationY;
				final int origLocX;
				final int origLocY = 0;
				
				//
				// render img and (if applicable) imgSpare
				//
				
				if (imgSpare == IMG_NONE) {
					origLocX = panel.getWidth() / 2 - img.getWidth() / 2;
				}
				else if (img.getWidth() + imgSpare.getWidth() <= panel.getWidth()) {
					int paddingRemaining = panel.getWidth() - (img.getWidth() + imgSpare.getWidth());
					
					g.drawImage(imgSpare, img.getWidth() + 2*paddingRemaining/3, 0, null);

					origLocX = paddingRemaining/3;
				}
				else {
					g.drawImage(imgSpare, img.getWidth(), 0, null);
					
					origLocX = 0;
				}
				
				g.drawImage(img, origLocX, origLocY, null);
				locationY = Math.max(img.getHeight(), imgSpare.getHeight()) + 2*pad;
				
				//
				// show all sub images that contain detected features
				//
				
				for (RectLifespan rect_life: keyPointDescriptorList.values()) {
					int x, y, w, h;
					long lifespan, id;
					if (rect_life != null) {
						x = rect_life.rect.x;
						y = rect_life.rect.y;
						w = rect_life.rect.width;
						h = rect_life.rect.height;
						id = rect_life.getID();
						lifespan = rect_life.lifespan;
					}
					else {
						lifespan = id = -1;
						x = y = 0;
						w = h = 1;
					}
					
					g.setColor(Color.magenta);
					g.drawRect(origLocX + x, origLocY + y, w, h);
					g.drawString("" + id, origLocX + x, origLocY + y - 2);
					

					double sc = (double) sz / Math.max(w, h);
					int w_ = (int) (sc * w);
					int h_ = (int) (sc * h);
					
					g.setColor(Color.black);
					g.drawString(x + "," + y + " | " + w + "x" + h + " | " + (lifespan - System.currentTimeMillis()), locationX, locationY-2);
					g.drawImage(img.getSubimage(x, y, w, h).getScaledInstance(w_, h_, BufferedImage.SCALE_FAST), locationX, locationY, null);

					locationX += sz + pad;
					if (locationX + sz > panel.getWidth()) {
						locationX = 0;
						locationY += sz + 2*pad;
					}
				}
				
				
				//
				// render metadata
				//
				
				frameCounter++;
				if (frameCounter % frameBatch == 0) {
					frameCounter = 0;
					avgFrameTime = (System.currentTimeMillis() - lastFrame) / frameBatch;
					lastFrame = System.currentTimeMillis();
				}
				g.setColor(Color.magenta);
				g.drawString("" + avgFrameTime, 0, 15);
				g.drawString(img.getWidth() + "x" + img.getHeight(), 0, 30);
				g.drawString(keyPointDescriptorList.size() + " boxes", 0, 45);
			}
		};
		frame.getContentPane().add(panel);
		
		
		new Timer(FRAME_TIME, (action) -> {
			imgSpare = IMG_NONE;
			
			try {
				
				runCascade();
//				runFindText();
//				runNN();
				
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			frame.repaint();
		}).start();
	}
	
	
	static Net net = null;
	static String nnDir = "res/nn/" +
	//////////////////////
	// Super Resolution //
	//////////////////////
	// EDSR (performance model, slow inference)
//			"EDSR_x2.pb"
//			"EDSR_x3.pb"
//			"EDSR_x4.pb"

	// ESPCN (speed model, real-time inference)
//			"ESPCN_x2.pb"
//			"ESPCN_x3.pb"
//			"ESPCN_x4.pb"

	// FSRCNN (speed model, real-time inference)
//			"FSRCNN_x2.pb"
//			"FSRCNN_x3.pb"
//			"FSRCNN_x4.pb"

	// LapSRN (balanced model, high scale factor)
//			"LapSRN_x2.pb"
//			"LapSRN_x4.pb"
//			"LapSRN_x8.pb"
	
	////////////////////
	// Classification //
	////////////////////
	// Caffe* (append prototxt / caffemodel when reading) 
			"bvlc_googlenet."
	;
	public static void runNN() throws Exception {
		if (net == null) {
			net = Dnn.readNet(nnDir + "prototxt", nnDir + "caffemodel");//, superResolutionDir + "txt");
			
//			capture = new VideoCapture();
//			capture.open(0);
			
			frame.setTitle(nnDir);
//			frame.setVisible(true);
		}
		
		
//		capture.read(mat);
		
//		java.awt.Point mouse = MouseInfo.getPointerInfo().getLocation();
//		mat = bimgToMat(rob.createScreenCapture(new Rectangle(mouse.x - 50,mouse.y - 50, 50 * 3, 50 * 3)));
		
		mat = Imgcodecs.imread("res/baboon.png");
		
		
		Mat blob = Dnn.blobFromImage(mat, 1.0f, new Size(224, 224), new Scalar(104, 117, 123), false, false);
		net.setInput(blob);
		Mat output = net.forward();
		
//		MinMaxLocResult res = Core.minMaxLoc(output.reshape(1, 1));
//		int classID = (int) res.minLoc.x;
		
		
		img = matToBimg(output);
		System.exit(0);
	}
	
	
	public static void runFindText() throws Exception {
		if (capture == null) {
			capture = new VideoCapture();
			capture.open(0);
			
			frame.setTitle("Finding text");
			frame.setVisible(true);
		}
		
		
		capture.read(mat);
		Mat proc = new Mat();
		
		Imgproc.cvtColor(mat, proc, Imgproc.COLOR_RGB2GRAY);
		Imgproc.GaussianBlur(proc, proc, new Size(5, 5), 0);
		
        Imgproc.morphologyEx(proc, proc, Imgproc.MORPH_GRADIENT, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(6, 3)));
        Imgproc.threshold(proc, proc, 0.0, 255.0, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        Imgproc.morphologyEx(proc, proc, Imgproc.MORPH_CLOSE, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 1)));
        

		
		Mat hierarchy = new Mat();
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(proc, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		

		ArrayList<Rect> boundsNew = new ArrayList<Rect>();
		Scalar rectCol = new Scalar(0, 255, 0);
		Scalar contourCol = new Scalar(255, 255, 255);
		for (int idx = 0; idx < contours.size(); idx++) {
			Rect rect = Imgproc.boundingRect(contours.get(idx));

			if (rect.width <= 16 || rect.height <= 16)
				continue;
			
			Imgproc.drawContours(proc, contours, idx, contourCol, Core.FILLED);
			Mat maskROI = proc.submat(rect);
			int r = 100 * Core.countNonZero(maskROI) / (rect.width * rect.height);

//			Imgproc.rectangle(mat, rect, rectCol, 1);
			if (r > 40) {
				boundsNew.add(rect);
			}
		}
		

		Rect[] boundsNew_ = new Rect[boundsNew.size()];
		boundsNew.toArray(boundsNew_);
		setBoundingBoxes(boundsNew_);
		
		
//		img = matToBimg(proc);
		img = matToBimg(mat);
	}

	static CascadeClassifier detector = null;
	static String featureCassifierDir = "res/cascades/" +
	// HAAR cascades
//			"haarcascade_frontalface_default.xml" // bad
//			"haarcascade_frontalface_alt.xml" // bad
			"haarcascade_frontalface_alt2.xml" // ok
//			"haarcascade_frontalface_alt_tree.xml" // ok
//			"haarcascade_profileface.xml" // good

//			"haarcascade_eye.xml" // ok
//			"haarcascade_eye_tree_eyeglasses.xml" // ok
//			"haarcascade_lefteye_2splits.xml" // ok
//			"haarcascade_righteye_2splits.xml" // ok

//			"haarcascade_smile.xml" // not working

//			"haarcascade_upperbody.xml" // bad
//			"haarcascade_lowerbody.xml" // ok
//			"haarcascade_fullbody.xml" // ok

//			"haarcascade_frontalcatface.xml" // ok
//			"haarcascade_frontalcatface_extended.xml" // good

//			"haarcascade_licence_plate_rus_16stages.xml" // not working
//			"haarcascade_russian_plate_number.xml" // bad

	// HOG cascades
//			"hogcascade_pedestrians.xml" // bad
	
	// LBP cascades
//			"lbpcascade_frontalface.xml" // ok
//			"lbpcascade_frontalface_improved.xml" // bad
//			"lbpcascade_profileface.xml" // good
//			"lbpcascade_frontalcatface.xml" // good
//			"lbpcascade_silverware.xml" // ok
	;
	public static void runCascade() throws Exception {
		if (detector == null) {
			detector = new CascadeClassifier();
			detector.load(featureCassifierDir);
			
			capture = new VideoCapture();
			capture.open(0);
			
			frame.setTitle(featureCassifierDir);
			frame.setVisible(true);
		}
		
		if (!capture.read(mat))
			return;
		
		Mat proc = new Mat();
		Imgproc.cvtColor(mat, proc, Imgproc.COLOR_BGR2GRAY);
		Imgproc.equalizeHist(proc, proc);
		

//		int minSize = Math.min(10, proc.rows());

		MatOfRect detectedBounds = new MatOfRect();
		detector.detectMultiScale(proc, detectedBounds);//, 1.2, 3, Objdetect.CASCADE_SCALE_IMAGE, new Size(minSize, minSize));
		Rect[] detectedRects = detectedBounds.toArray();

		Scalar boundCol = new Scalar(0, 255, 0);
		for (Rect rect: detectedRects)
			Imgproc.rectangle(mat, rect, boundCol, 1);

		setBoundingBoxes(detectedRects);
		
		img = matToBimg(mat);
	}
	
	
	static ORB featureDetector = ORB.create();
	static BFMatcher featureMatcher = BFMatcher.create(Core.NORM_HAMMING, true);
	private static void setBoundingBoxes(Rect[] boundsNew) {
		if (boundsNew != null) {
			for (int i = 0; i < boundsNew.length && keyPointDescriptorList.size() < 100; ++i) {
				Mat subMat = mat.submat(boundsNew[i]);
				MatOfKeyPoint keyPointMat = new MatOfKeyPoint();
				Mat desc = new Mat();
				
				featureDetector.detect(subMat, keyPointMat);
				featureDetector.compute(subMat, keyPointMat, desc);
				
				
				boolean featuresRecognised = false;
				
				for (Mat keyPointDescriptor: keyPointDescriptorList.keySet()) {
					MatOfDMatch matches = new MatOfDMatch();
					featureMatcher.match(desc, keyPointDescriptor, matches);
					if (!matches.empty() && Core.countNonZero(matches) > Core.countNonZero(keyPointDescriptor) * .9) { // 90% match is acceptable
						keyPointDescriptorList.get(keyPointDescriptor).rect = boundsNew[i];
						keyPointDescriptorList.get(keyPointDescriptor).resetLifespan();
						
						featuresRecognised = true;
						
						break;
					}
				}
				
				if (!featuresRecognised) {
					keyPointDescriptorList.put(keyPointMat, new RectLifespan(boundsNew[i]));
				}
			}
		}
		
		/*
		if (boundsNew != null) {
			for (Rect rect: boundsNew) {
				boundingBoxes.add(boundsNew);
			}
		}
		
		ArrayList<Point> centers = new ArrayList<Point>();
		ArrayList<RectLifespan> rects = new ArrayList<RectLifespan>();
		for (int i = keyPointDescriptorList.size() - 1; i >= 0; --i) {
			RectLifespan current = boundingBoxes.get(i);
			Point currentCenter = new Point(current.rect.x + current.rect.width/2, current.rect.y + current.rect.height/2);

			if (current.lifespan <= System.currentTimeMillis()) {
				boundingBoxes.remove(i);
				continue;
			}

			for (int j = 0; j < centers.size(); ++j) {
				RectLifespan other = rects.get(j);
				Point otherCenter = centers.get(j);
				
				if (current.rect.contains(otherCenter) || other.rect.contains(currentCenter)) {
					current.rect = other.rect;
					current.resetLifespan();
					
					boundingBoxes.remove(other);
					rects.remove(j);
					centers.remove(j);
				}
			}
			
			current.assignID();
			centers.add(currentCenter);
			rects.add(current);
		}*/
		
		for (Mat desc: keyPointDescriptorList.keySet()) {
			RectLifespan rect_life = keyPointDescriptorList.get(desc);
			if (rect_life == null) {
				
			}

			if (rect_life.lifespan <= System.currentTimeMillis()) {
				keyPointDescriptorList.put(desc, null);
			}
		}
	}
	
	public static BufferedImage matToBimg(Mat matrix) throws Exception {
		if (matrix.empty())
			return new BufferedImage(matrix.width(), matrix.height(), BufferedImage.TYPE_INT_RGB);
			
		MatOfByte mob = new MatOfByte();
		Imgcodecs.imencode(".jpg", matrix, mob);
		return ImageIO.read(new ByteArrayInputStream(mob.toArray()));
	}

	public static Mat bimgToMat(BufferedImage img) {
		Mat mat = new Mat(img.getHeight(), img.getWidth(), CvType.CV_32S);
		mat.put(0, 0, ((DataBufferInt) img.getRaster().getDataBuffer()).getData());
		return mat;
	}
}

class RectLifespan {
	public static final long LIFESPAN_DEFAULT = 2000; // in milliseconds
	public long lifespan;
	
	public Rect rect;
	
	private static long ID = 0;
	private long id = -1;
	
	
	public RectLifespan(Rect rect) {
		this.rect = rect;
		resetLifespan();
		
		assignID(); // NOTE: temp - added to ensure rects in OpenCVTester#keyPointDescriptorList have an id, remove if map is no longer used 
	}
	
	public void resetLifespan() {
		lifespan = System.currentTimeMillis() + LIFESPAN_DEFAULT;
	}
	
	public long getID() {
		return id;
	}
	
	public void assignID() {
		if (id == -1)
			id = ID++;
	}
}