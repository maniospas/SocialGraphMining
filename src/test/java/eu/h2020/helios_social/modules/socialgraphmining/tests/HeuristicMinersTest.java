package eu.h2020.helios_social.modules.socialgraphmining.tests;

import org.junit.Test;

import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RepeatAndReplyMiner;
import org.junit.Assert;

public class HeuristicMinersTest extends BaseMinerTestFunctionalities {
	
	@Override
	public TestDevice getDevice(String name) {
		TestDevice device = super.getDevice(name);
		device.getMiner().setActiveMiner("repeat");
		return device;
	}

	@Test(expected = Exception.class)
	public void shouldExpectCEN() {
		Utils.development = true;
		new RepeatAndReplyMiner(null);
	}
	
	@Test(expected = Exception.class)
	public void shouldRecommendOnlyForContext() {
        getDevice("A").getMiner().recommendInteractions(null);
	}
	
	@Test
	public void shouldAlwaysRecommendTheMostRecentInteraction() {
		getDevice("A").send(getDevice("B"));
		getDevice("C").send(getDevice("A"));
		getDevice("A").send(getDevice("D"));
		getDevice("D").send(getDevice("B"));
		getDevice("D").send(getDevice("E"));
		getDevice("E").send(getDevice("B"));
		getDevice("B").send(getDevice("A"));
		
		Assert.assertEquals(argmax(getDevice("A").recommendInteractionsInCurrentContext()), "B");
		Assert.assertEquals(argmax(getDevice("D").recommendInteractionsInCurrentContext()), "E");
	}
	
	@Test
	public void shouldNotHaveProblemWithRemovedCENNodes() {
		getDevice("A").send(getDevice("B"));
		getDevice("A").getMiner().getContextualEgoNetwork().removeNodeIfExists("B");
		getDevice("A").recommendInteractionsInCurrentContext();
	}
}
