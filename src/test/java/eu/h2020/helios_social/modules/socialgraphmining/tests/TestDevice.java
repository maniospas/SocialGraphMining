package eu.h2020.helios_social.modules.socialgraphmining.tests;

import java.util.ArrayList;
import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.storage.NoStorage;
import eu.h2020.helios_social.modules.socialgraphmining.SwitchableMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNMiner;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RepeatAndReplyMiner;

public class TestDevice {
	private ContextualEgoNetwork contextualEgoNetwork;
	private SwitchableMiner miner;
	
	public TestDevice(String name) {
		contextualEgoNetwork = ContextualEgoNetwork.createOrLoad(new NoStorage("NOFILESYSTEM\\"), name, null);
		SwitchableMiner miner = new SwitchableMiner(contextualEgoNetwork);
		miner.createMiner("repeat", RepeatAndReplyMiner.class);
		miner.createMiner("gnn", GNNMiner.class).setDeniability(0, 0);
		this.miner = miner;
		miner.setActiveMiner("gnn");
		contextualEgoNetwork.setCurrent(contextualEgoNetwork.getOrCreateContext("default"));
	}
	public SwitchableMiner getMiner() {
		return miner;
	}
	public String getName() {
		return contextualEgoNetwork.getEgo().getId();
	}
	public void send(TestDevice other) {
		Interaction interaction = contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(other.getName(), null))
				.addDetectedInteraction(null);
		other.receive(this, miner.getModelParameters(interaction));
	}
	protected void receive(TestDevice other, String parameters) {
		Interaction interaction = contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getOrCreateNode(other.getName(), null), contextualEgoNetwork.getEgo())
				.addDetectedInteraction(null);
		miner.newInteraction(interaction, parameters, InteractionType.RECEIVE);
		other.receiveAck(this, miner.getModelParameters(interaction));
	}
	protected void receiveAck(TestDevice other, String parameters) {
		ArrayList<Interaction> interactions = contextualEgoNetwork
				.getCurrentContext()
				.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(other.getName(), null))
				.getInteractions();
		miner.newInteraction(interactions.get(interactions.size()-1), parameters, InteractionType.RECEIVE_REPLY);
	}
	public HashMap<Node, Double> recommendInteractionsInCurrentContext() {
		return miner.recommendInteractions(miner.getContextualEgoNetwork().getCurrentContext());
	}

}
