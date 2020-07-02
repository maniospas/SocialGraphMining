package eu.h2020.helios_social.modules.socialgraphmining.examples;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNMiner;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.modules.socialgraphmining.SwitchableMiner;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RepeatAndReplyMiner;

public class TestCase {
	public static class Device {
		private ContextualEgoNetwork contextualEgoNetwork;
		private HashSet<Device> neighbors = new HashSet<Device>();
		private SocialGraphMiner miner;
		
		public Device(String name) {
			contextualEgoNetwork = ContextualEgoNetwork.createOrLoad("experiment_data\\", name, null);
			
			
			SwitchableMiner miner = new SwitchableMiner(contextualEgoNetwork);
			miner.createMiner("repeat", RepeatAndReplyMiner.class);
			miner.createMiner("gnn", GNNMiner.class).setDeniability(0.1, 0.1);
			
			
			this.miner = miner;
			//this.miner = new AdditionalDiscoveryMiner(miner, miner.getMiner("repeat"), 3);
			
			miner.setActiveMiner("gnn");
			contextualEgoNetwork.setCurrent(contextualEgoNetwork.getOrCreateContext("default"));
		}
		public String getName() {
			return contextualEgoNetwork.getEgo().getId();
		}
		public void send(Device other) {
			Interaction interaction = contextualEgoNetwork
					.getCurrentContext()
					.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(other.getName(), null))
					.addDetectedInteraction(null);
			other.receive(this, miner.getModelParameters(interaction));
			/*neighbors.add(other);
			for(Device neighbor : neighbors)
			if(neighbor!=other){
				Interaction interaction = neighbor.contextualEgoNetwork
						.getCurrentContext()
						.getOrAddEdge(neighbor.contextualEgoNetwork.getOrCreateNode(other.getName(), null), 
								neighbor.contextualEgoNetwork.getOrCreateNode(getName(), null))
						.addDetectedInteraction(null);
				neighbor.miner.newInteraction(interaction, miner.getModelParameters(null), InteractionType.RECEIVE);
			}*/
		}
		protected void receive(Device other, String parameters) {
			Interaction interaction = contextualEgoNetwork
					.getCurrentContext()
					.getOrAddEdge(contextualEgoNetwork.getOrCreateNode(other.getName(), null), contextualEgoNetwork.getEgo())
					.addDetectedInteraction(null);
			miner.newInteraction(interaction, parameters, InteractionType.RECEIVE);
			other.receiveAck(this, miner.getModelParameters(interaction));
		}
		protected void receiveAck(Device other, String parameters) {
			ArrayList<Interaction> interactions = contextualEgoNetwork
					.getCurrentContext()
					.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(other.getName(), null))
					.getInteractions();
			miner.newInteraction(interactions.get(interactions.size()-1), parameters, InteractionType.RECEIVE_REPLY);
		}
		public Node recommendNextInteraction() {
			HashMap<Node, Double> interactionScores = miner.recommendInteractions(contextualEgoNetwork.getCurrentContext());
			double bestScore = 0;
			Node bestNode = null;
			for(Entry<Node, Double> interactionScore : interactionScores.entrySet()) 
				if(interactionScore.getValue() > bestScore) {
					bestScore = interactionScore.getValue();
					bestNode = interactionScore.getKey();
				}
			return bestNode;
		}
	}
	
	
	public static void main(String[] args) throws Exception {
		Device A = new Device("A");
		Device B = new Device("B");
		A.send(B);
	}
}
