import numpy as np
from bilby.core.prior import Prior, Uniform, ConditionalBeta


class MinimumPrior(ConditionalBeta):
    def __init__(self, order, duration, minimum=0, maximum=1, name=None,
                 latex_label=None, unit=None, boundary=None):
        super(MinimumPrior, self).__init__(
            alpha=1, beta=order, minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label, unit=unit,
            boundary=boundary, condition_func=self.minimum_condition
        )
        self.duration = duration
        self.order = order
        self.reference_name = self.name[:-1] + str(int(self.name[-1]) - 1)
        self._required_variables = [self.reference_name]

    def minimum_condition(self, reference_params, **kwargs):
        return dict(minimum=kwargs[self.reference_name] + 1 / duration)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)


class SpikeAndSlab(Prior):
    def __init__(self, slab=None, mix=0.5, name=None, latex_label=None, unit=None):
        """Spike and slab with spike at the slab minimum

        Parameters
        ----------

        """
        if isinstance(slab, Uniform) is False:
            raise NotImplementedError()
        minimum = slab.minimum
        maximum = slab.maximum
        super(SpikeAndSlab, self).__init__(
            name=name, latex_label=latex_label, unit=unit, minimum=minimum,
            maximum=maximum)
        self.mix = mix
        self.spike = minimum
        self.slab = slab

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate SpikeAndSlab prior.

        Parameters
        ----------
        val: Union[float, int, array_like]

        This maps to the inverse CDF. This has been analytically solved for this case,

        """
        self.test_valid_for_rescaling(val)

        if isinstance(val, (float, int)):
            p = (val - self.mix) / (1 - self.mix)
            if p < 0:
                icdf = self.minimum
            else:
                icdf = self.minimum + p * (self.maximum - self.minimum)
        else:
            p = (val - self.mix) / (1 - self.mix)
            icdf = self.minimum + p * (self.maximum - self.minimum)
            icdf[p < 0] = self.minimum

        return icdf

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        float: Prior probability of val
        """

        if isinstance(val, (float, int)):
            if val == self.spike:
                return self.mix
            else:
                return (1 - self.mix) * self.slab.prob(val)
        else:
            probs = self.slab.prob(val) * (1 - self.mix)
            probs[val == self.spike] = self.mix
            return probs
