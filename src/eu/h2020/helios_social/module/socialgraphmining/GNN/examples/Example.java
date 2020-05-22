package eu.h2020.helios_social.module.socialgraphmining.GNN.examples;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.module.socialgraphmining.GNN.GNNMiner;

public class Example {
	public static class Device {
		private ContextualEgoNetwork contextualEgoNetwork;
		private SocialGraphMiner miner;
		public Device(String name) {
			contextualEgoNetwork = ContextualEgoNetwork.createOrLoad("data\\", name, null);
			miner = new GNNMiner(contextualEgoNetwork);
			contextualEgoNetwork.setCurrent(contextualEgoNetwork.getOrCreateContext("default"));
		}
		
		public String getName() {
			return contextualEgoNetwork.getEgo().getId();
		}
		
		public void send(Device other) {
			other.receive(this, miner.getModelParameters(null));
		}
		public void receive(Device other, String parameters) {
			Interaction interaction = contextualEgoNetwork
					.getCurrentContext()
					.getOrAddEdge(contextualEgoNetwork.getOrCreateNode(other.getName(), null), contextualEgoNetwork.getEgo())
					.addDetectedInteraction(null);
			miner.newInteraction(interaction, parameters, InteractionType.RECEIVE);
			other.receiveAck(this, miner.getModelParameters(interaction));
		}
		protected void receiveAck(Device other, String parameters) {
			Interaction interaction = contextualEgoNetwork
					.getCurrentContext()
					.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(other.getName(), null))
					.addDetectedInteraction(null);
			System.out.println("other "+other.getName());
			System.out.println("other "+contextualEgoNetwork.getOrCreateNode(other.getName(), null).getId());
			miner.newInteraction(interaction, parameters, InteractionType.RECEIVE_REPLY);
		}
	}
	
	
	public static void main(String[] args) {
		Device deviceA = new Device("userA");
		Device deviceB = new Device("userB");
		Device deviceC = new Device("userC");
		Device deviceD = new Device("userD");
		Device deviceE = new Device("userE");
		
		deviceA.send(deviceB);
		deviceA.send(deviceC);
		deviceB.send(deviceA);
		deviceD.send(deviceC);
		deviceC.send(deviceE);
		deviceC.send(deviceD);
		deviceC.send(deviceA);
		
		deviceA.contextualEgoNetwork.save();
	}
}
