// import java.util.ArrayList;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.FileWriter;
import java.io.BufferedWriter;


import java.util.Arrays;

public class Train {
	public static double[] AmulA (double[] a0, double[] a1) {
		double[] res = new double[a0.length];
		for (int i=0;i<a0.length;i++) {
			res[i] = a0[i] * a1[i];
		}
		return res;
	}
	public static double[] AsubA (double[] a0, double[] a1) {
		double[] res = new double[a0.length];
		for (int i=0;i<a0.length;i++) {
			res[i] = a0[i] - a1[i];
		}
		return res;
	}
	public static double sumA (double[] a0) {
		double res = 0.0d;
		for (int i=0;i<a0.length;i++){
			res += a0[i];
		}
		return res;
	}

	public static double lossOf (double[] prm) {	// 损失函数
		/* 共 500 人，取前 200 个男性和前 200 个女性用来训练，剩下 100 个人用来测试 */
		double ls = 0;
		for (double[] s : train_m) {
			double tp = sumA(AmulA(prm, s));
			tp = 1 - 1.0d / (1.0d + Math.exp(- tp));
			// tp = Math.tan(tp);
			ls += tp;
		}
		for (double[] s : train_f) {
			double tp = sumA(AmulA(prm, s));
			tp = 1.0d / (1.0d + Math.exp(- tp));
			// tp = Math.tan(tp);
			ls += tp;
		}
		return ls / train_m.length;
	}
	public static void move (double step) {
		// read
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
		// calc
		double loss_now = lossOf(prm);
		double[] d = new double[prm.length];	// 求导
		for (int i=0;i<prm.length;i++) {
			double[] q = prm.clone();
			q[i] += step;
			d[i] = (lossOf(q) - loss_now) / step;
		}
		
		ogPrmTxt = "";
		for (int i=0;i<prm.length;i++) {
			prm[i] = prm[i] - step*d[i];
			ogPrmTxt += prm[i] + " ";
		}
		ogPrmTxt = ogPrmTxt.trim();
		// save
		try {
			BufferedWriter prmW = new BufferedWriter(new FileWriter(new File("target.txt")));
			prmW.write(ogPrmTxt);
			prmW.flush();
			prmW.close();
		} catch (Exception e) {}


		System.out.print("loss = ");
		System.out.print(loss_now);
		System.out.println();
		return;
	}
	public static double[][] train_m;
	public static double[][] train_f;
	public static double[][] test_m;
	public static double[][] test_f;
	public static void main (String[] args) {
		String ftFolderName = "../data/imgFeature";
		File ftFolder = new File(ftFolderName);	// 特征文件夹 File 对象
		String[] ftList = ftFolder.list();	// 获取文件夹中所有文件名
		double[][] fts = new double[500][513];
		for (int i=0;i<ftList.length;i+=2) {
			int j = i / 2;
			String feaName = ftFolderName + "/" + ftList[i];	// 特征.txt 文件路径
			String ogFeaTxt = "";
			try {
				FileInputStream fis = new FileInputStream(new File(feaName));
				InputStreamReader isr = new InputStreamReader(fis);
				BufferedReader feaR = new BufferedReader(isr);
				ogFeaTxt = feaR.readLine() + ",1";
			} catch (Exception e) {}
			String[] feaTxts = ogFeaTxt.split(",");
			double[] f = new double[513];	// 特征数组
			for (int k=0;k<feaTxts.length;k++) {
				f[k] = Double.parseDouble(feaTxts[k]);
			}
			fts[j] = f;
		}
		// fts[i] 代表第 i+1 张图片的特征数组 [513]
		train_m = Arrays.copyOfRange(fts, 0  , 220);
		train_f = Arrays.copyOfRange(fts, 250, 470);
		test_m = Arrays.copyOfRange(fts, 220, 250);
		test_f = Arrays.copyOfRange(fts, 470, 500);
		// System.out.println();

		int n = 50;
		while (n-- > 0) {
			move(0.02);
			System.out.println(n);
		}
	}
}