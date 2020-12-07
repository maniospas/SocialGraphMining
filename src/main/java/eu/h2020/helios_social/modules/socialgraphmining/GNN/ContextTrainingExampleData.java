package eu.h2020.helios_social.modules.socialgraphmining.GNN;

import java.util.ArrayList;

import mklab.JGNN.core.Tensor;

/**
 * This class provides a storage structure that organizes a list of {@link TrainingExample} data to be stored in the 
 * contextual ego network's contexts.
 * 
 * It is indented to be used as a dynamically created instance on contexts (which are cross module components)
 * by calling <code>context.getOrCreateInstance(GNNNodeData.class)</code> to either retrieve of create it.
 * 
 * @author Emmanouil Krasanakis
 */
public class ContextTrainingExampleData {
	private ArrayList<TrainingExample> trainingExamples = null;
	public Tensor transformToSrcEmbedding = null;
	public Tensor transformToDstEmbedding = null;
	
	public ContextTrainingExampleData() {}
	
	public synchronized void addTrainingExample(TrainingExample example) {
		/*ArrayList<TrainingExample> trainingExamples = getTrainingExampleList();
		for(TrainingExample prevExample : trainingExamples) {
			if(prevExample.getSrc()==example.getSrc() && prevExample.getDst()==example.getDst() && prevExample.getLabel()==example.getLabel()) {
				prevExample.impulseWeight();
				return;
			}
		}*/
		trainingExamples.add(example);
	}
	
	/**
	 * Grants direct access to a list of training examples to traverse or edit.
	 * @return An array list of training examples.
	 */
	public synchronized ArrayList<TrainingExample> getTrainingExampleList() {
		if(trainingExamples==null)
			trainingExamples = new ArrayList<TrainingExample>();
		return trainingExamples;
	}
	
	/**
	 * Calls the {@link TrainingExample#degrade} operation for each {@link TrainingExample} in the
	 * data (e.g. to reduce all weights) and removes those that end up on a very small weight.
	 * @param factor The degrading factor passed on to the degrade operation.
	 * @param removalThreshold The threshold weight under which training examples are removed.
	 * @see TrainingExample#getWeight()
	 */
	public synchronized void degrade(double factor, double removalThreshold) {
		for(TrainingExample trainingExample : new ArrayList<TrainingExample>(getTrainingExampleList())) {
			trainingExample.degrade(factor);
			if(trainingExample.getWeight() < removalThreshold)
				trainingExamples.remove(trainingExample);
		}
	}
}
