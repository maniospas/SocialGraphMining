package eu.h2020.helios_social.module.socialgraphmining;

import eu.h2020.helios_social.core.contextualegonetwork.Node;

/**
 * Provides an abstraction of evaluation measures.
 * 
 * @author Emmanouil Krasanakis
 */
public abstract class Measure {
	/**
	 * Supervised evaluation of interactions send by each node.
	 * @param socialGraphMiner The social graph miner used to perform the prediction.
	 * @param dst The destination node.
	 * @return The evaluation score - typically a value in the range [0,1]
	 */
	public abstract double evaluateSend(SocialGraphMiner socialGraphMiner, Node dst);
}
