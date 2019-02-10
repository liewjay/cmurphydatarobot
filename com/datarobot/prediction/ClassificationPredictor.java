package com.datarobot.prediction;

import com.datarobot.prediction.BaseClassificationPredictor;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.io.Serializable;

public class ClassificationPredictor extends BaseClassificationPredictor
		implements
			Serializable {

	private static final String modelId = "5c5df0267c6f8b2418b6bc07";

	protected ClassificationPredictor(MulticlassPredictor predictor) {
		super(predictor);
	}

	/**
	 * Returns an instance of the loaded ClassificationPredictor If more than
	 * one model is loaded, use
	 * {@link BaseClassificationPredictor#getPredictor(String)} instead
	 * <p>
	 * (If more than one model is loaded, the model that will be loaded first by
	 * the Java ClassLoader will be returned. In such a scenario, the behavior
	 * of this method is undefined)
	 * 
	 * @return
	 */
	public static ClassificationPredictor getPredictor()
			throws IllegalAccessException, InstantiationException,
			ClassNotFoundException {
		return getPredictor(modelId);
	}

	/**
	 * Returns an instance of the loaded BaseRegressionPredictor by model ID
	 * 
	 * @param id
	 * @return
	 */
	public static ClassificationPredictor getPredictor(String id)
			throws ClassNotFoundException, IllegalAccessException,
			InstantiationException {
		return new ClassificationPredictor(
				(MulticlassPredictor) BaseClassificationPredictor.getModel(id));
	}
}