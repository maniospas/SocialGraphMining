package eu.h2020.helios_social.modules.socialgraphmining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.modules.socialgraphmining.examples.Example.Device;

/**
 * This class can be used to wrap {@link SocialGraphMiner} instances at the top level so that interactions can be sent and received
 * 
 * @author Emmanouil Krasanakis
 */
public class SocialGraphMinerManager {
	private SocialGraphMiner miner;
	
	protected SocialGraphMinerManager(SocialGraphMiner miner) {
		this.miner = miner;
	}
	
	/**
	 * Should be called when an 
	 * @param alterId The node id of the alter to which.
	 * @return The parameters to send alongside the interaction.
	 */
	public String send(String alterId) {
		Interaction interaction = miner.getContextualEgoNetwork()
				.getCurrentContext()
				.getOrAddEdge(miner.getContextualEgoNetwork().getEgo(), miner.getContextualEgoNetwork().getOrCreateNode(alterId, null))
				.addDetectedInteraction(null);
		return miner.getModelParameters(interaction);
	}
	
	/**
	 * 
	 * @param alterId The node if of the alter from which
	 * @param parameters
	 * @return
	 */
	public String receive(String alterId, String parameters) {
		Interaction interaction = miner.getContextualEgoNetwork()
				.getCurrentContext()
				.getOrAddEdge(miner.getContextualEgoNetwork().getOrCreateNode(alterId, null), miner.getContextualEgoNetwork().getEgo())
				.addDetectedInteraction(null);
		miner.newInteraction(interaction, parameters, InteractionType.RECEIVE);
		return miner.getModelParameters(interaction);
	}
	
	public void receiveAck(String alterId, String parameters) {
		ArrayList<Interaction> interactions = miner.getContextualEgoNetwork()
				.getCurrentContext()
				.getOrAddEdge(miner.getContextualEgoNetwork().getEgo(), miner.getContextualEgoNetwork().getOrCreateNode(alterId, null))
				.getInteractions();
		miner.newInteraction(interactions.get(interactions.size()-1), parameters, InteractionType.RECEIVE_REPLY);
	}
	
	/**
	 * 
	 * @return
	 */
	public SocialGraphMiner getMiner() {
		return miner;
	}
	
}
