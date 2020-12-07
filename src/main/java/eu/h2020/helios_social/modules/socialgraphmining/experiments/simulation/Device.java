package eu.h2020.helios_social.modules.socialgraphmining.experiments.simulation;

import java.util.ArrayList;
import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;

/**
 * This class provide the abstraction of a device that performs the communication protocol
 * needed by a {@link SocialGraphMiner} in a simulated communication setting. Instances of
 * this class are created by {@link Simulation} for experiments over the efficacy of different
 * types of miners.
 * 
 * @author Emmanouil Krasanakis
 */
public class Device {
	private SocialGraphMiner miner;
	
	public Device(SocialGraphMiner miner) {
		this.miner = miner;
		miner.getContextualEgoNetwork().setCurrent(miner.getContextualEgoNetwork().getOrCreateContext("default"));
	}
	public SocialGraphMiner getMiner() {
		return miner;
	}
	public String getName() {
		return miner.getContextualEgoNetwork().getEgo().getId();
	}
	public void send(Device other) {
		Interaction interaction = miner.getContextualEgoNetwork()
				.getCurrentContext()
				.getOrAddEdge(miner.getContextualEgoNetwork().getEgo(), miner.getContextualEgoNetwork().getOrCreateNode(other.getName(), null))
				.addDetectedInteraction(null);
		other.receive(this, miner.getModelParameters(interaction));
	}
	protected void receive(Device other, String parameters) {
		Interaction interaction = miner.getContextualEgoNetwork()
				.getCurrentContext()
				.getOrAddEdge(miner.getContextualEgoNetwork().getOrCreateNode(other.getName(), null), miner.getContextualEgoNetwork().getEgo())
				.addDetectedInteraction(null);
		miner.newInteraction(interaction, parameters, InteractionType.RECEIVE);
		other.receiveAck(this, miner.getModelParameters(interaction));
	}
	protected void receiveAck(Device other, String parameters) {
		ArrayList<Interaction> interactions = miner.getContextualEgoNetwork()
				.getCurrentContext()
				.getOrAddEdge(miner.getContextualEgoNetwork().getEgo(), miner.getContextualEgoNetwork().getOrCreateNode(other.getName(), null))
				.getInteractions();
		miner.newInteraction(interactions.get(interactions.size()-1), parameters, InteractionType.RECEIVE_REPLY);
	}
	public HashMap<Node, Double> recommendInteractionsInCurrentContext() {
		return miner.recommendInteractions(miner.getContextualEgoNetwork().getCurrentContext());
	}

}
