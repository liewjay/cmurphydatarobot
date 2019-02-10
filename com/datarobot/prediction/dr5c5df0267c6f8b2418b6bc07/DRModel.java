/*-----------------------------------------------------------------------------------------------
    Copyright @2017. DataRobot, Inc. All Rights Reserved. Permission to use, copy, modify,
    and distribute this software and its documentation is hereby granted, provided that the
    above copyright notice, this paragraph and the following two paragraphs appear in all copies,
    modifications, and distributions of this software or its documentation. Contact DataRobot,
    1 International Place, 5th Floor, Boston, MA, United States 02110, support@datarobot.com
    for more details.

    IN NO EVENT SHALL DATAROBOT BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
    OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS OR LOST DATA, ARISING OUT OF THE USE OF THIS
    SOFTWARE AND ITS DOCUMENTATION BASED ON ANY THEORY OF LIABILITY, EVEN IF DATAROBOT HAS BEEN
    ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS".
    DATAROBOT SPECIFICALLY DISCLAIMS ANY AND ALL WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  DATAROBOT HAS NO
    OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *----------------------------------------------------------------------------------------------*/

package com.datarobot.prediction.dr5c5df0267c6f8b2418b6bc07;
// imports

import java.io.DataInputStream;
import java.util.concurrent.CountDownLatch;
import java.io.ByteArrayInputStream;
import java.util.Comparator;
import java.util.Vector;
import java.util.ArrayList;
import java.io.Serializable;
import com.datarobot.prediction.Row;
import com.datarobot.prediction.MulticlassPredictor;
import java.io.DataInput;
import java.util.Collections;
import java.io.ObjectOutputStream;
import com.datarobot.drmatrix.DoubleArray;
import java.util.concurrent.Callable;
import java.util.HashMap;
import java.lang.reflect.Method;
import java.io.ObjectInputStream;
import com.datarobot.drmatrix.DenseDoubleArray;
import java.util.LinkedHashMap;
import java.io.DataOutputStream;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Map;
import java.io.File;
import java.util.concurrent.Future;
import java.io.IOException;
import java.io.InvalidClassException;
import java.util.HashMap;
import com.datarobot.drmatrix.DoubleArray;
import java.io.File;
import com.datarobot.drmatrix.DoubleArray;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.List;
import java.io.ByteArrayOutputStream;
import com.datarobot.drmatrix.SparseDoubleArray;
import java.io.InputStream;
import java.util.concurrent.Executors;

// global helpers

class DRDataInputStream extends DataInputStream {
	private byte buffer[] = new byte[65535];

	public DRDataInputStream(InputStream is) {
		super(is);
	}

	public String readString() throws IOException {
		int len = readInt();
		// reallocate buffer if insufficient
		if (len > this.buffer.length)
			this.buffer = new byte[len * 2];
		int read = 0;
		while (read < len)
			read += read(this.buffer, read, len - read);
		return new String(this.buffer, 0, len, "UTF-8");
	}

}

abstract class AbstractDRTask implements Serializable {
	public void readParameters(String resourcename) throws Exception {
		InputStream is = null;
		try {
			is = getClass().getResourceAsStream(resourcename);
			readParameters(new DRDataInputStream(is));
		} finally {
			if (is != null)
				is.close();
		}
	}

	public abstract void readParameters(DRDataInputStream dis) throws Exception;

	protected void validatedSerialVersionUID(long expected, DataInputStream dis)
			throws IOException, InvalidClassException, Exception {
		validatedSerialVersionUID(expected, dis.readLong());
	}

	protected void validatedSerialVersionUID(long expected, long found)
			throws InvalidClassException {
		if (expected != found)
			throw new InvalidClassException(getClass().getName(),
					"The serialized parameters (" + found
							+ ") and the class implementation (" + expected
							+ ") does not match!");
	}
}

// data structures
class BaseDataStructure {
	public boolean uses_single_precision = false;
	public DoubleArray d = null;
	public String[] s = null;
	public double offset = 0.0;
	public double exposure = 1.0;
	public double predictions_to_boost = 0.0;
	public double forecast_distance = 0.0;

	public void copy(BaseDataStructure x) {
		this.uses_single_precision = x.uses_single_precision;
		this.offset = x.offset;
		this.exposure = x.exposure;
		this.predictions_to_boost = x.predictions_to_boost;
		this.forecast_distance = x.forecast_distance;
		if (x.d != null) {
			this.d = x.d.copy();
		}
		if (x.s != null) {
			this.s = new String[x.s.length];
			System.arraycopy(x.s, 0, this.s, 0, x.s.length);
		}
	}
}

class BaseVertex implements Serializable {
	private static final long serialVersionUID = 6882110001L;

	// task profiling
	BaseVertex subtasks[] = null;
	Object submodels[] = null;

	protected void execute(BaseDataStructure in, BaseDataStructure out)
			throws Exception {
	}

	public void run(BaseDataStructure in, BaseDataStructure out)
			throws Exception {
		execute(in, out);
	}

	public static double[] normalize(double[] x) {
		double[] out = new double[x.length];
		double sum = 0;
		for (int i = 0; i < x.length; i++)
			sum += x[i];
		if (sum != 0) {
			sum = 1.0 / sum;
			for (int i = 0; i < x.length; i++)
				out[i] = x[i] * sum;
		}
		return out;
	}

	interface Estimator {

		public double predict(BaseDataStructure in) throws Exception;
		public double predict_proba(BaseDataStructure in) throws Exception;
		public double[] predict_multi(BaseDataStructure in, int n_classes)
				throws Exception;
		public double predict_margin(BaseDataStructure in) throws Exception;
		public double[] predict_margin_multi(BaseDataStructure in, int n_classes)
				throws Exception;
	}

	abstract class BaseEstimator extends AbstractDRTask implements Estimator {
		protected boolean is_multiclass;
		protected boolean fit_skipped;
		protected int num_fit_classes;
		protected int[] missing_classes_idx;

		abstract public double predict(BaseDataStructure in) throws Exception;

		@Override
		public double predict_margin(BaseDataStructure in) throws Exception {
			return this.predict(in);
		}

		@Override
		public double predict_proba(BaseDataStructure in) throws Exception {
			return this.predict(in);
		}

		@Override
		protected void validatedSerialVersionUID(long expected,
				DataInputStream dis) throws IOException, InvalidClassException,
				Exception {
			super.validatedSerialVersionUID(expected, dis);
			this.is_multiclass = dis.readBoolean();
			if (this.is_multiclass) {
				// read missing classes
				this.read_missing_classes((DRDataInputStream) dis);
			}
		}

		@Override
		public double[] predict_multi(BaseDataStructure in, int n_classes)
				throws Exception {
			double scores[] = new double[1];
			scores[0] = this.predict(in);
			return scores;
		}

		protected double[] predict_fit_skipped(BaseDataStructure in,
				int n_classes) throws Exception {
			double scores[] = new double[n_classes];
			scores[0] = 1;
			return this.handle_missclass_output(this.missing_classes_idx,
					scores);
		}

		@Override
		public double[] predict_margin_multi(BaseDataStructure in, int n_classes)
				throws Exception {
			double[] scores;
			if (this.fit_skipped) {
				scores = this.predict_fit_skipped(in, n_classes);
			} else {
				scores = this.predict_multi(in, n_classes);
			}

			// apply logit
			for (int i = 0; i < scores.length; i++)
				scores[i] = logit(scores[i]);
			return scores;
		}

		protected double[] handle_missclass_output(int[] missclass_idx,
				double[] proba_out) {
			if (missclass_idx == null || missclass_idx.length == 0) {
				return proba_out;
			}
			double[] proba = new double[proba_out.length + missclass_idx.length];
			int missing_idx = 0;
			int score_idx = 0;
			for (int i = 0; i < proba.length; i++) {
				if (missing_idx < missclass_idx.length
						&& missclass_idx[missing_idx] == i) {
					missing_idx++;
				} else {
					proba[i] = proba_out[score_idx];
					score_idx++;
				}
			}
			return proba;
		}

		protected void read_missing_classes(DRDataInputStream dis)
				throws Exception {
			this.fit_skipped = dis.readBoolean();
			this.num_fit_classes = dis.readInt();
			int missing_classes_n = dis.readInt();
			this.missing_classes_idx = readIntArray(dis, missing_classes_n);
		}
	}

	class PDM3 extends AbstractDRTask implements Transformer {
		private static final long serialVersionUID = 6882110037L;
		private int[] feature_idx;
		private String[] feature_names;
		private StrIntHashMap lookup;
		private String separator;

		public PDM3() {

		}

		@Override
		public void readParameters(DRDataInputStream dis) throws Exception {
			this.validatedSerialVersionUID(serialVersionUID, dis);
			int i;
			int maplen = 0;

			int feature_len = dis.readInt();
			this.feature_idx = new int[feature_len];
			this.feature_names = new String[feature_len];
			for (i = 0; i < feature_len; i++) {
				this.feature_idx[i] = dis.readInt();
				this.feature_names[i] = dis.readString();
			}

			this.lookup = new StrIntHashMap();
			maplen = dis.readInt();
			for (i = 0; i < maplen; i++) {
				String key = dis.readString();
				this.lookup.put(key, dis.readInt());
			}

			// read separator
			this.separator = dis.readString();
		}

		@Override
		public void transform(BaseDataStructure in, BaseDataStructure out) {
			int in_idx = 0;
			int out_idx = 0;
			String level;
			out.d.zero();
			for (int i = 0; i < this.feature_idx.length; i++) {
				in_idx = this.feature_idx[i];
				// replace missing values with nan string
				level = in.s[in_idx];
				if (level.equals(""))
					level = "nan";
				out_idx = lookUpStrIntHashMap(this.lookup,
						this.feature_names[i] + this.separator + level, -1);
				if (out_idx != -1)
					out.d.set(out_idx, 1);
				else {
					out_idx = lookUpStrIntHashMap(this.lookup,
							this.feature_names[i] + this.separator
									+ "small_count", -1);
					if (out_idx != -1)
						out.d.set(out_idx, 1);
				}
			}
		}
	}

	class StrIntHashMap extends HashMap<String, Integer> {
	}
	public static int lookUpStrIntHashMap(StrIntHashMap map, String key,
			int replacement) {
		Integer value = map.get(key);
		if (value == null)
			value = replacement;
		return value;
	}

	public static String getResourcePath(Object instance, String name) {
		Class clazz = instance.getClass();
		String pkgName = clazz.getPackage().getName().substring(25)
				.replace(".", "/");
		StringBuilder stringBuilder = new StringBuilder();
		String parts[] = {pkgName, clazz.getSimpleName(), name};
		for (String s : parts) {
			stringBuilder.append("/");
			stringBuilder.append(s);
		}
		return stringBuilder.toString();
	}

	interface Modeler {

		public void predict(BaseDataStructure in, BaseDataStructure out)
				throws Exception;
		public void predict_margin(BaseDataStructure in, BaseDataStructure out)
				throws Exception;
		public void transform(BaseDataStructure in, BaseDataStructure out)
				throws Exception;
	}

	public static double logit(double x) {
		return Math.log(x / (1.0 - x));
	}

	class BaseModeler extends AbstractDRTask implements Modeler {
		private static final long serialVersionUID = 6882110062L;
		int scores_dims;
		float clip_threshold;
		boolean classification;
		boolean add_exposure;
		boolean clip_offset;
		boolean predictions_to_boost;
		int[] use_cols;
		int[] feature_selection_cols;
		double right_cens;
		double left_cens;
		BaseEstimator estimator;

		public BaseModeler() {

		}

		@Override
		public void readParameters(DRDataInputStream dis) throws Exception {
			this.validatedSerialVersionUID(serialVersionUID, dis);

			this.add_exposure = dis.readBoolean();
			this.clip_offset = dis.readBoolean();
			this.clip_threshold = dis.readFloat();
			this.use_cols = readIntArray(dis, -1);
			this.left_cens = dis.readDouble();
			this.right_cens = dis.readDouble();
			this.predictions_to_boost = dis.readBoolean();
			this.feature_selection_cols = readIntArray(dis, -1);

			this.scores_dims = dis.readInt();
			if (this.scores_dims >= 0) {
				this.classification = dis.readBoolean();

				Class parent_cls = Class.forName(this.getClass().getPackage()
						.getName()
						+ ".BaseVertex");
				Object parent_instance = parent_cls.newInstance();

				// instantiate Estimator
				String estClassname = dis.readString();
				Class<?> cls = Class.forName(this.getClass().getPackage()
						.getName()
						+ ".BaseVertex$" + estClassname);

				this.estimator = (BaseEstimator) cls.getDeclaredConstructor(
						parent_cls).newInstance(parent_instance);
				((AbstractDRTask) this.estimator).readParameters(dis);

			} else {
				this.use_cols = new int[0];
			}
		}

		void initialize_input(BaseDataStructure in) {
			if (this.add_exposure)
				in.offset += Math.log(in.exposure);
			if (this.clip_offset)
				in.offset = Math.max(Math.min(in.offset, this.clip_threshold),
						-this.clip_threshold);

			if (this.use_cols.length > 0) {
				DoubleArray selected_input;
				if (in.d instanceof SparseDoubleArray)
					selected_input = new SparseDoubleArray(this.use_cols.length);
				else
					selected_input = new DenseDoubleArray(this.use_cols.length);
				for (int i = 0; i < this.use_cols.length; i++)
					selected_input.set(i, in.d.get(this.use_cols[i]));
				in.d = selected_input;
			}
		}

		BaseDataStructure prep_apply(BaseDataStructure in) throws Exception {
			BaseDataStructure copy = in.getClass().newInstance();
			copy.copy(in);
			return copy;
		}

		private double cap_prediction(double value) {
			// cap predictions
			if (!Double.isNaN(this.right_cens))
				value = Math.min(this.right_cens, value);
			if (!Double.isNaN(this.left_cens))
				value = Math.max(this.left_cens, value);
			return value;
		}

		@Override
		public void predict(BaseDataStructure in, BaseDataStructure out)
				throws Exception {

			in = prep_apply(in);
			initialize_input(in);

			if (this.predictions_to_boost) {
				if (this.classification)
					out.predictions_to_boost = cap_prediction(this.estimator
							.predict_proba(in));
				else
					out.predictions_to_boost = cap_prediction(this.estimator
							.predict(in));
			} else {
				if (this.classification) {
					if (this.estimator.is_multiclass) {
						double scores[];
						if (this.estimator.fit_skipped) {
							scores = this.estimator.predict_fit_skipped(in,
									this.scores_dims);
						} else {
							scores = this.estimator.predict_multi(in,
									this.scores_dims);
						}
						for (int i = 0; i < out.d.size(); i++)
							out.d.set(i, cap_prediction(scores[i]));
					} else {
						out.d.set(
								0,
								cap_prediction(this.estimator.predict_proba(in)));
					}
				} else
					out.d.set(0, cap_prediction(this.estimator.predict(in)));
			}
		}

		@Override
		public void predict_margin(BaseDataStructure in, BaseDataStructure out)
				throws Exception {

			in = prep_apply(in);
			initialize_input(in);

			if (this.predictions_to_boost) {
				out.d.set(0, 0);
				out.predictions_to_boost = this.estimator.predict_margin(in);
			} else {
				if (this.estimator.is_multiclass) {
					double scores[] = this.estimator.predict_margin_multi(in,
							this.scores_dims);
					for (int i = 0; i < out.d.size(); i++)
						out.d.set(i, scores[i]);
				} else {
					out.d.set(0, this.estimator.predict_margin(in));
				}
			}
		}

		@Override
		public void transform(BaseDataStructure in, BaseDataStructure out)
				throws Exception {
			if (this.feature_selection_cols.length == 0)
				return;
			int[] selected_features = this.use_cols.length > 0
					? this.use_cols
					: this.feature_selection_cols;
			for (int i = 0; i < selected_features.length; i++)
				out.d.set(i, in.d.get(selected_features[i]));
		}
	}

	class Linear extends BaseEstimator {
		private static final long serialVersionUID = 6882110027L;
		private short loss;
		private double intercept[];
		private double intercept_scaling;
		private DoubleArray coefficients[];
		private double tweedie_p;
		private boolean is_adagrad;
		private boolean ova;

		public Linear() {

		}

		@Override
		public void readParameters(DRDataInputStream dis) throws Exception {
			this.validatedSerialVersionUID(serialVersionUID, dis);

			if (this.fit_skipped)
				return;

			this.loss = dis.readShort();
			this.tweedie_p = Double.longBitsToDouble(dis.readLong());
			this.intercept = readDoubleArray(dis, -1);
			int num_class = dis.readInt();
			int num_coeffs = dis.readInt();
			this.coefficients = new DoubleArray[num_class];
			for (int i = 0; i < num_class; i++) {
				this.coefficients[i] = new SparseDoubleArray(num_coeffs);
				for (int j = 0; j < num_coeffs; j++) {
					double v = Double.longBitsToDouble(dis.readLong());
					if (v != 0)
						this.coefficients[i].set(j, v);
				}
			}
			this.intercept_scaling = Double.longBitsToDouble(dis.readLong());
			this.is_adagrad = dis.readBoolean();
			this.ova = dis.readBoolean();
		}

		public int get_input_size() {
			return this.coefficients[0].size();
		}

		public double[] score(DoubleArray in_d, int output_len)
				throws Exception {
			double[] scores = new double[Math.max(this.intercept.length,
					output_len)];
			for (int i = 0; i < this.intercept.length; i++)
				scores[i] = (in_d.dot(this.coefficients[i]) + this.intercept[i]
						* this.intercept_scaling);
			return scores;
		}

		@Override
		public double predict_margin(BaseDataStructure in) throws Exception {
			return this.score(in.d, 1)[0];
		}

		@Override
		public double predict(BaseDataStructure in) throws Exception {
			double s = this.score(in.d, 1)[0];
			s += in.offset;
			switch (this.loss) {
				case 1 : // log
					s = logistic(s);
					break;
				case 2 : // modified huber
					s = (Math.min(1, Math.max(-1, s)) + 1) / 2;
					break;
				case 4 : // poisson
				case 5 : // gamma
					if (s > 700.0)
						s = 700.0;
					s = Math.exp(s);
					break;
			}
			return s;
		}

		@Override
		public double[] predict_multi(BaseDataStructure in, int n_classes)
				throws Exception {
			double scores[] = this.score(in.d, n_classes);
			switch (this.loss) {
				case 1 : // log
					if (this.num_fit_classes <= 2) {
						scores[1] = logistic(scores[0]);
						scores[0] = 1.0 - scores[1];
					} else {
						if (this.ova || !this.is_adagrad) {
							for (int i = 0; i < scores.length; i++)
								scores[i] = logistic(scores[i]);
							scores = normalize(scores);
						} else {
							scores = softmax(scores);
						}
					}
					break;
			}
			return this.handle_missclass_output(this.missing_classes_idx,
					scores);
		}
	}

	static int[] readIntArray(DataInput in, int len) throws IOException {
		if (len == -1)
			len = in.readInt();
		int[] out = new int[len];
		for (int i = 0; i < len; i++)
			out[i] = in.readInt();
		return out;
	}

	static short[] readShortArray(DataInput in, int len) throws IOException {
		if (len == -1)
			len = in.readInt();
		short[] out = new short[len];
		for (int i = 0; i < len; i++)
			out[i] = in.readShort();
		return out;
	}

	static double[] readDoubleArray(DataInput in, int len) throws IOException {
		if (len == -1)
			len = in.readInt();
		double[] out = new double[len];
		for (int i = 0; i < len; i++)
			out[i] = Double.longBitsToDouble(in.readLong());
		return out;
	}

	static float[] readFloatArray(DataInput in, int len) throws IOException {
		if (len == -1)
			len = in.readInt();
		float[] out = new float[len];
		for (int i = 0; i < len; i++)
			out[i] = Float.intBitsToFloat(in.readInt());
		return out;
	}

	interface Transformer {

		public void transform(BaseDataStructure in, BaseDataStructure out)
				throws Exception;
	}

	static void vec_copy(DoubleArray src, int src_start, DoubleArray dest,
			int dest_start, int len, boolean src_uses_single_precision,
			boolean dest_uses_single_precision) {
		if (src_uses_single_precision || dest_uses_single_precision) {
			for (int i = 0; i < len; i++) {
				dest.set(dest_start + i,
						(double) ((float) src.get(src_start + i)));
			}
		} else {
			for (int i = 0; i < len; i++) {
				dest.set(dest_start + i, src.get(src_start + i));
			}
		}
	}

	static void vec_copy(DoubleArray src, int src_start, DoubleArray dest,
			int dest_start, int len) {
		vec_copy(src, src_start, dest, dest_start, len, false, false);
	}

	class PNI2 extends AbstractDRTask implements Transformer {
		private static final long serialVersionUID = 6882110040L;
		private boolean single_precision;
		private boolean round_scaling;
		private int[] selected;
		private short[] fillIndicators;
		private double[] medians;
		private double[] ranges;
		private static final double SINGLE_PRECISION = 3.40282355e38;

		public PNI2() {

		}

		@Override
		public void readParameters(DRDataInputStream dis) throws Exception {
			this.validatedSerialVersionUID(serialVersionUID, dis);
			this.single_precision = dis.readBoolean();
			this.round_scaling = dis.readBoolean();
			int len = dis.readInt();
			this.selected = readIntArray(dis, len);
			this.fillIndicators = readShortArray(dis, len);
			this.medians = readDoubleArray(dis, len);
			this.ranges = readDoubleArray(dis, len);
		}

		@Override
		public void transform(BaseDataStructure in, BaseDataStructure out)
				throws Exception {
			int in_idx = 0;
			int out_idx = 0;
			for (int i = 0; i < this.selected.length; i++) {
				in_idx = this.selected[i];
				// median impute
				double v = in.d.get(in_idx);
				if (this.single_precision)
					v = (float) v;
				if (Double.isNaN(v))
					v = this.medians[i];
				else {
					// cap extreme values if single precision expected
					if (this.single_precision) {
						if (v > SINGLE_PRECISION)
							v = 1e35;
						else if (v < -SINGLE_PRECISION)
							v = -1e35;
					}
				}

				// scale small
				if (this.ranges[i] <= 0.1 && this.ranges[i] != 0)
					if (this.single_precision) {
						if (this.round_scaling)
							v *= Math.pow(10.0f, (int) (Math
									.log10(1.0f / (float) this.ranges[i])) + 1);
						else
							v = (float) v / (float) this.ranges[i];
					} else {
						if (this.round_scaling)
							v *= Math
									.pow(10.0, (int) (Math
											.log10(1.0 / this.ranges[i])) + 1);
						else
							v /= this.ranges[i];
					}

				out.d.set(out_idx, v);

				// fill indicator
				if (fillIndicators[i] > 0)
					out.d.set(++out_idx, Double.isNaN(in.d.get(in_idx))
							? 1.0
							: 0.0);

				out_idx++;
			}
		}
	}

	class FeatureHandler extends AbstractDRTask {
		private static final long serialVersionUID = 6882110019L;
		private int in_double_idx[];
		private int out_double_idx[];
		private int in_string_idx[];
		private int out_string_idx[];

		public FeatureHandler() {

		}

		@Override
		public void readParameters(DRDataInputStream dis) throws Exception {
			this.validatedSerialVersionUID(serialVersionUID, dis);
			this.in_double_idx = readIntArray(dis, -1);
			this.out_double_idx = readIntArray(dis, -1);
			this.in_string_idx = readIntArray(dis, -1);
			this.out_string_idx = readIntArray(dis, -1);
			dis.close();
		}

		void process(BaseDataStructure in, BaseDataStructure out) {
			// perform feature assignments
			for (int i = 0; i < this.in_double_idx.length; i++)
				out.d.set(this.out_double_idx[i],
						in.d.get(this.in_double_idx[i]));
			for (int i = 0; i < this.in_string_idx.length; i++)
				out.s[this.out_string_idx[i]] = in.s[this.in_string_idx[i]];
		}
	}

	public static double[] softmax(double x[]) {
		double sum = 0;
		double max = x[0];
		for (int i = 0; i < x.length; i++) {
			if (x[i] > max)
				max = x[i];
		}
		for (int i = 0; i < x.length; i++) {
			x[i] = (float) Math.exp(x[i] - max);
			sum += x[i];
		}
		for (int i = 0; i < x.length; i++)
			x[i] /= sum;
		return x;
	}

	public static float[] softmax(float x[]) {
		float sum = 0;
		float max = x[0];
		for (int i = 0; i < x.length; i++) {
			if (x[i] > max)
				max = x[i];
		}
		for (int i = 0; i < x.length; i++) {
			x[i] = (float) Math.exp(x[i] - max);
			sum += x[i];
		}
		for (int i = 0; i < x.length; i++)
			x[i] /= sum;
		return x;
	}

	public static double logistic(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}
	public static float logistic(float x) {
		return 1.0f / (1.0f + (float) Math.exp(-x));
	}

	class RST extends AbstractDRTask implements Transformer {
		private static final long serialVersionUID = 6882110047L;
		private DoubleArray median;
		private DoubleArray mad;

		public RST() {

		}

		@Override
		public void readParameters(DRDataInputStream dis) throws Exception {
			this.validatedSerialVersionUID(serialVersionUID, dis);

			double[] medianArray = readDoubleArray(dis, -1);
			this.median = new DenseDoubleArray(medianArray.length);
			for (int i = 0; i < medianArray.length; i++)
				this.median.set(i, medianArray[i]);

			double[] madArray = readDoubleArray(dis, -1);
			this.mad = new DenseDoubleArray(madArray.length);
			for (int i = 0; i < madArray.length; i++)
				this.mad.set(i, madArray[i]);
		}

		public void transform(BaseDataStructure in, BaseDataStructure out) {
			if (out.d != in.d)
				out.d.set(in.d);
			out.d.subtract(this.median);
			out.d.divide(this.mad);
		}
	}

	public static void nan_to_num(DoubleArray x, double nan_value,
			boolean single_precision) {
		double maxf, minf;
		if (single_precision) {
			maxf = 3.4028235e+38;
			minf = -3.4028235e+38;
		} else {
			maxf = 1.7976931348623157e+308;
			minf = -1.7976931348623157e+308;
		}

		double[] data = null;
		if (x instanceof SparseDoubleArray)
			data = ((SparseDoubleArray) x).getData();
		else
			data = ((DenseDoubleArray) x).getData();

		for (int i = 0; i < data.length; i++) {
			if (Double.isNaN(data[i]))
				data[i] = nan_value;
			else if (Double.isInfinite(data[i]))
				data[i] = data[i] > 0 ? maxf : minf;
		}
	}

	public BaseVertex() throws Exception {
	}
}

class BP_I extends BaseDataStructure {
	public BP_I() {

		this.d = new DenseDoubleArray(13);
		this.s = new String[1];
	}
}
class V1_I extends BaseDataStructure {
	public V1_I() {

		this.d = new DenseDoubleArray(0);
		this.s = new String[1];
	}
}
class V2_I extends BaseDataStructure {
	public V2_I() {

		this.d = new DenseDoubleArray(13);
		this.s = new String[0];
	}
}
class V3_I extends BaseDataStructure {
	public V3_I() {

		this.d = new DenseDoubleArray(14);
		this.s = new String[0];
		this.uses_single_precision = true;
	}
}
class V4_I extends BaseDataStructure {
	public V4_I() {

		this.d = new SparseDoubleArray(15);
		this.s = new String[0];
		this.uses_single_precision = true;
	}
}

class DP_O extends BaseDataStructure {
	public DP_O() {

		this.d = new DenseDoubleArray(13);
		this.s = new String[1];
	}
}
class V1_O extends BaseDataStructure {
	public V1_O() {

		this.d = new SparseDoubleArray(1);
		this.s = new String[0];
		this.uses_single_precision = true;
	}
}
class V2_O extends BaseDataStructure {
	public V2_O() {

		this.d = new DenseDoubleArray(14);
		this.s = new String[0];
		this.uses_single_precision = true;
	}
}
class V3_O extends BaseDataStructure {
	public V3_O() {

		this.d = new SparseDoubleArray(14);
		this.s = new String[0];
		this.uses_single_precision = true;
	}
}
class BP_O extends BaseDataStructure {
	public BP_O() {

		this.d = new DenseDoubleArray(1);
		this.s = new String[0];
	}
}

// task codes
class DP extends BaseVertex {
	// sparse i/o?
	boolean sparse_in = false;
	boolean sparse_out = false;
	// task helpers

	final private static String[] naValues = {"null", "na", "n/a", "#N/A",
			"N/A", "?", ".", "Inf", "INF", "inf", "-inf", "-Inf", "-INF",
			"-1.#IND", "1.#QNAN", "1.#IND", "-1.#QNAN", "#N/A N/A", "NA",
			"#NA", "NULL", "NaN", "nan", " ", "None"};

	final private StrIntHashMap naMap = new StrIntHashMap() {
		{
			for (int i = 0; i < naValues.length; i++)
				this.put(naValues[i], 1);
		}
	};

	final FeatureHandler feature_handler = new FeatureHandler();

	public DP() throws Exception {

		feature_handler.readParameters(getResourcePath(this,
				"feature_assignments"));

	}
	// task action
	protected void execute(BaseDataStructure in, BaseDataStructure out)
			throws Exception {

		// remove pre-defined NA values
		if (in.s != null) {
			for (int i = 0; i < in.s.length; i++) {
				if (this.naMap.get(in.s[i]) != null)
					in.s[i] = "";
			}
		}

		this.feature_handler.process(in, out);

	}
}
class V1 extends BaseVertex {
	// sparse i/o?
	boolean sparse_in = false;
	boolean sparse_out = true;
	// task helpers

	private final PDM3 transformer = new PDM3();

	public V1() throws Exception {

		this.transformer.readParameters(getResourcePath(this, "PDM3"));

	}
	// task action
	protected void execute(BaseDataStructure in, BaseDataStructure out)
			throws Exception {

		DoubleArray original_in_d = in.d;
		String[] original_in_s = in.s;

		this.transformer.transform(in, out);

		for (int i = 0; i < out.d.size(); i++)
			out.d.set(i, (float) out.d.get(i));

		if (out.d != null)
			nan_to_num(out.d, 0, out.uses_single_precision);

		in.d = original_in_d;
		in.s = original_in_s;

	}
}
class V2 extends BaseVertex {
	// sparse i/o?
	boolean sparse_in = false;
	boolean sparse_out = false;
	// task helpers

	private final PNI2 transformer = new PNI2();

	public V2() throws Exception {

		this.transformer.readParameters(getResourcePath(this, "PNI2"));

	}
	// task action
	protected void execute(BaseDataStructure in, BaseDataStructure out)
			throws Exception {

		DoubleArray original_in_d = in.d;
		String[] original_in_s = in.s;

		this.transformer.transform(in, out);

		for (int i = 0; i < out.d.size(); i++)
			out.d.set(i, (float) out.d.get(i));

		if (out.d != null)
			nan_to_num(out.d, 0, out.uses_single_precision);

		in.d = original_in_d;
		in.s = original_in_s;

	}
}
class V3 extends BaseVertex {
	// sparse i/o?
	boolean sparse_in = false;
	boolean sparse_out = true;
	// task helpers

	private final RST transformer = new RST();

	public V3() throws Exception {

		this.transformer.readParameters(getResourcePath(this, "RST"));

	}
	// task action
	protected void execute(BaseDataStructure in, BaseDataStructure out)
			throws Exception {

		DoubleArray original_in_d = in.d;
		String[] original_in_s = in.s;

		this.transformer.transform(in, out);

		for (int i = 0; i < out.d.size(); i++)
			out.d.set(i, (float) out.d.get(i));

		if (out.d != null)
			nan_to_num(out.d, 0, out.uses_single_precision);

		in.d = original_in_d;
		in.s = original_in_s;

	}
}
class V4 extends BaseVertex {
	// sparse i/o?
	boolean sparse_in = true;
	boolean sparse_out = false;
	// task helpers

	private final BaseModeler modeler = new BaseModeler();

	public V4() throws Exception {

		this.modeler.readParameters(getResourcePath(this, "BaseModeler"));

	}
	// task action
	protected void execute(BaseDataStructure in, BaseDataStructure out)
			throws Exception {

		DoubleArray original_in_d = in.d;
		String[] original_in_s = in.s;

		this.modeler.predict(in, out);

		if (out.d != null)
			nan_to_num(out.d, 0, out.uses_single_precision);

		in.d = original_in_d;
		in.s = original_in_s;

	}
}
class BP extends BaseVertex {
	// sparse i/o?
	boolean sparse_in = false;
	boolean sparse_out = false;
	// task helpers

	final String[] double_predictors;
	final String[] string_predictors;

	public String[] get_double_predictors() {
		return this.double_predictors;
	}

	public String[] get_string_predictors() {
		return this.string_predictors;
	}

	DP dp = new DP();
	V1 v1 = new V1();
	V2 v2 = new V2();
	V3 v3 = new V3();
	V4 v4 = new V4();
	public BP() throws Exception {

		DRDataInputStream dis = new DRDataInputStream(getClass()
				.getResourceAsStream(
						getResourcePath(this, "codegen_input_predictors")));

		// read predictor names
		this.double_predictors = new String[dis.readInt()];
		for (int i = 0; i < double_predictors.length; i++)
			this.double_predictors[i] = dis.readString();
		this.string_predictors = new String[dis.readInt()];
		for (int i = 0; i < string_predictors.length; i++)
			this.string_predictors[i] = dis.readString();

		this.subtasks = new BaseVertex[5];

		this.subtasks[0] = this.dp;

		this.subtasks[1] = this.v1;

		this.subtasks[2] = this.v2;

		this.subtasks[3] = this.v3;

		this.subtasks[4] = this.v4;

	}
	// task action
	protected void execute(BaseDataStructure in, BaseDataStructure out)
			throws Exception {
		DP_O dp_out = new DP_O();
		dp.run(in, dp_out);
		V1_O v1_out = new V1_O();
		V1_I v1_in = new V1_I();
		System.arraycopy(dp_out.s, 0, v1_in.s, 0, 1);
		v1.run(v1_in, v1_out);
		V2_O v2_out = new V2_O();
		V2_I v2_in = new V2_I();

		vec_copy(dp_out.d, 0, v2_in.d, 0, 13, dp_out.uses_single_precision,
				v2_in.uses_single_precision);

		v2.run(v2_in, v2_out);
		V3_O v3_out = new V3_O();
		V3_I v3_in = new V3_I();

		vec_copy(v2_out.d, 0, v3_in.d, 0, 14, v2_out.uses_single_precision,
				v3_in.uses_single_precision);
		v3.run(v3_in, v3_out);
		V4_I v4_in = new V4_I();

		vec_copy(v1_out.d, 0, v4_in.d, 0, 1, v1_out.uses_single_precision,
				v4_in.uses_single_precision);

		vec_copy(v3_out.d, 0, v4_in.d, 1, 14, v3_out.uses_single_precision,
				v4_in.uses_single_precision);
		v4.run(v4_in, out);
	}
}

public class DRModel implements MulticlassPredictor, Serializable {
	private final int version = 1;
	private final String metadata = "{\"lid\": \"5c5df0267c6f8b2418b6bc07\", \"blueprint\": {\"1\": [[\"CAT\"], [\"PDM3 cm=50000;dtype=float32;sc=10\"], \"T\"], \"3\": [[\"2\"], [\"RST dtype=float32\"], \"T\"], \"2\": [[\"NUM\"], [\"PNI2 dtype=float32\"], \"T\"], \"4\": [[\"1\", \"3\"], [\"LRCD \"], \"P\"]}, \"pid\": \"5c5deda31b17bd442b943fa4\"}";

	BP task = new BP();

	private String[] classLabels = {"1", "0"};

	@Override
	public String[] get_double_predictors() {
		return task.get_double_predictors();
	}

	@Override
	public String[] get_string_predictors() {
		return task.get_string_predictors();
	}

	@Override
	public double score(Row r) throws Exception {
		return score_multi(r)[0];
	}

	private double[] score_multi(Row r) throws Exception {
		BP_I in = new BP_I();
		BP_O out = new BP_O();

		if (r.d.length > 0)
			for (int i = 0; i < in.d.size(); i++)
				in.d.set(i, r.d[i]);

		if (r.s.length > 0)
			System.arraycopy(r.s, 0, in.s, 0, r.s.length);
		task.run(in, out);

		return ((DenseDoubleArray) out.d).getData();
	}

	public Map<String, Double> classificationScore(Row r) throws Exception {
		Map<String, Double> scores = new LinkedHashMap<String, Double>();
		double positiveProb = score(r);
		double negativeProb = 1.0 - positiveProb;
		if (positiveProb >= negativeProb) {
			scores.put(classLabels[0], positiveProb);
			scores.put(classLabels[1], negativeProb);
		} else {
			scores.put(classLabels[1], negativeProb);
			scores.put(classLabels[0], positiveProb);
		}
		return scores;
	}

	public int getVersion() {
		return this.version;
	}

	public String getMetadata() {
		return this.metadata;
	}

	public DRModel() throws Exception {
	}

}