package com.datarobot.prediction;

import com.datarobot.batch.processor.BatchProcessor;

public class ModelApplication {
	public static void main(String[] args) {
		new BatchProcessor("dr5c5df0267c6f8b2418b6bc07").run(args);
	}
}