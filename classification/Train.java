import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.Arrays;

public class Train {
	final String train_m_folder = "../data/train_m_ft";
	final String train_f_folder = "../data/train_f_ft";
	public static double lossOf (double[] prm) {
		double ls = 0;
		double tp;
		double[] s = new double[train_m[0].length];
		for (int i=0;i<train_m.length;i++) {
			tp = 0;
			for (int j=0;j<prm.length;j++) {
				tp += prm[j] * train_m[i][j];
			}
			ls += 1 - 1.0d / (1.0d + Math.exp(- tp));
		}
		for (int i=0;i<train_f.length;i++) {
			tp = 0;
			for (int j=0;j<prm.length;j++) {
				tp += prm[j] * train_f[i][j];
			}
			ls += 1.0d / (1.0d + Math.exp(- tp));
		}	
		return ls / train_m.length;
	}
	public static void move (double step) {
		// read
		// double[] prm = readPrm();
		// calc
		double loss_now = lossOf(prm);
		double[] d = new double[prm.length];
		double[] q = new double[prm.length];
		for (int i=0;i<prm.length;i++) {
			System.arraycopy(prm, 0, q, 0, 513);
			q[i] += step;
			d[i] = (lossOf(q) - loss_now) / step;
		}
		for (int i=0;i<prm.length;i++) {
			prm[i] = prm[i] - step*d[i];
		}
		// savePrm(prm);
		System.out.print("loss = ");
		System.out.print(loss_now);
		System.out.println();
		return;
	}
	public static double[] readPrm () {
		String ogPrmTxt = "";
		try {
			FileInputStream fis = new FileInputStream(new File("target.txt"));
			InputStreamReader isr = new InputStreamReader(fis);
			BufferedReader prmR = new BufferedReader(isr);
			ogPrmTxt = prmR.readLine();
		} catch (Exception e) {}
		String[] prmTxts = ogPrmTxt.split(" ");
		double[] prm = new double[513];
		for (int i=0;i<prmTxts.length;i++) {
			prm[i] = Double.parseDouble(prmTxts[i]);
		}
		return prm;
	}
	public static void savePrm (double[] prm) {
		String ogPrmTxt = "";
		for (int i=0;i<prm.length;i++) {
			ogPrmTxt += prm[i] + " ";
		}
		ogPrmTxt = ogPrmTxt.trim();
		try {
			BufferedWriter prmW = new BufferedWriter(new FileWriter(new File("target.txt")));
			prmW.write(ogPrmTxt);
			prmW.flush();
			prmW.close();
		} catch (Exception e) {}
	}
	public static double[][] getfts (String folderPath) {
		File ftFolder = new File(folderPath);	// 文件夹 File 对象
		String[] ftList = ftFolder.list();	// 获取其中所有文件名
		double[][] fts = new double[ftList.length][513];
		for (int i=0;i<ftList.length;i+=1) {
			String feaName = folderPath + "/" + ftList[i];	//TODO
			String ogFeaTxt = "";	// 用于存放读取到的文件内容
			try {
				FileInputStream fis = new FileInputStream(new File(feaName));
				InputStreamReader isr = new InputStreamReader(fis);
				BufferedReader feaR = new BufferedReader(isr);
				ogFeaTxt = feaR.readLine() + " 1";
			} catch (Exception e) {}
			String[] feaTxts = ogFeaTxt.split(" ");	// 分割 为 513 个子串
			double[] f = new double[513];	// 特征数组
			for (int j=0;j<feaTxts.length;j++) {
				f[j] = Double.parseDouble(feaTxts[j]);
			}
			fts[i] = f;
		}
		return fts;
	};
	public static double[][] train_m;
	public static double[][] train_f;
	public static double[][] test_m;
	public static double[][] test_f;

	public static double[] prm;
	public static void main (String[] args) {

		// train_m = Arrays.copyOfRange(fts, 50, 220);
		train_f = getfts("../data/train_f_ft");
		train_m = getfts("../data/train_m_ft");

		train_m = Arrays.copyOfRange(train_m, 100, 500);
		train_f = Arrays.copyOfRange(train_f, 0, 450);

		prm = readPrm();
		int n = 5000000;
		// n = 10;
		while (n-- > 0) {
			move(0.03);
			if (n%20 == 0){
				System.out.println(n);
				savePrm(prm);
			}
		}
		savePrm(prm);
	}
}