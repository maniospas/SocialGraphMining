package eu.h2020.helios_social.module.socialgraphmining.GNN;

import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;

/**
 * A class models a training example used in GNN architectures that can be written
 * a tuplet (src, dst, weight, label). The source and destination node pairs
 * cannot be modeled through an Interaction of the contextual ego network module,
 * because they also include potentially negative examples.
 * 
 * @author Emmanouil Krasanakis
 */
public class TrainingExample {
	private Node src;
	private Node dst;
	private int label;
	private double weight;
	
	protected TrainingExample() {}
	
	public TrainingExample(Node src, Node dst, int label) {
		if(src==null || dst==null)
			Utils.error(new IllegalArgumentException());
		if(label!=0 && label!=1)
			Utils.error("Training example labels need be either 0 or 1");
		this.src = src;
		this.dst = dst;
		this.label = label;
		weight = 1;
	}
	
	/**
	 * Multiplies the weight of the training example with a given factor.
	 * @param factor A degrading factor of training example weight.
	 * @return <code>this</code> TrainingExample instance.
	 */
	public TrainingExample degrade(double factor) {
		weight *= factor;
		return this;
	}
	
	/**
	 * @return The source node of the interaction.
	 */
	public Node getSrc() {
		return src;
	}

	/**
	 * @return The destination node of the interaction.
	 */
	public Node getDst() {
		return dst;
	}
	
	/**
	 * @return A binary label of whether the training example is of an existing (1) or non-existing (0) interaction.
	 */
	public int getLabel() {
		return label;
	}
	
	/**
	 * @return The weight of the interaction.
	 * @see #degrade(double)
	 */
	public double getWeight() {
		return weight;
	}
}