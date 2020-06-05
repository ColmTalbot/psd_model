import numpy as np
from bilby.core.prior import Prior, Uniform, ConditionalBeta


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


class MinimumPrior(ConditionalBeta):

    def __init__(self, order, minimum=0, maximum=1, name=None,
                 latex_label=None, unit=None, boundary=None):
        super(MinimumPrior, self).__init__(
            alpha=1, beta=order, minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label, unit=unit,
            boundary=boundary, condition_func=self.minimum_condition
        )
        self.order = order
        self.reference_name = self.name[:-1] + str(int(self.name[-1]) - 1)
        self._required_variables = [self.reference_name]
        self.__class__.__name__ = 'MinimumPrior'

    def minimum_condition(self, reference_params, **kwargs):
        return dict(minimum=kwargs[self.reference_name])

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)


class MaximumPrior(ConditionalBeta):

    def __init__(self, order, minimum=0, maximum=1, name=None,
                 latex_label=None, unit=None, boundary=None):
        super(MaximumPrior, self).__init__(
            alpha=order, beta=1, minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label, unit=unit,
            boundary=boundary, condition_func=self.maximum_condition
        )
        self.order = order
        self.reference_name = self.name[:-1] + str(int(self.name[-1]) - 1)
        self.max_order_name = f"{self.name[:2]}_n_lorentzians"
        self._required_variables = [self.reference_name, self.max_order_name]
        self.__class__.__name__ = 'MaximumPrior'

    def maximum_condition(self, reference_params, **kwargs):
        return dict(
            maximum=kwargs[self.reference_name],
            alpha=kwargs[self.max_order_name] - self.order
        )

    def rescale(self, val, **required_variables):
        self.update_conditions(**required_variables)
        if self.alpha < 1:
            return self.minimum
        else:
            return super(MaximumPrior, self).rescale(val)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)


class VariableBeta(ConditionalBeta):

    def __init__(
        self, alpha=1, beta=1, minimum=0,
        maximum=1, name=None, latex_label=None, unit=None,
        boundary=None, alpha_parameter=None, beta_parameter=None
    ):
        super(VariableBeta, self).__init__(
            alpha=alpha, beta=beta, minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label, unit=unit,
            boundary=boundary, condition_func=self._condition
        )
        self.alpha_parameter = alpha_parameter
        self.beta_parameter = beta_parameter
        self._required_variables = [
            value for value in [alpha_parameter, beta_parameter]
            if value is not None
        ]
        self._variable_names = {
            name: value for name, value in zip(
                ["alpha", "beta"], [alpha_parameter, beta_parameter]
            )
            if value is not None
        }
        self.__class__.__name__ = 'VariableBeta'

    def _condition(self, reference_params, **kwargs):
        output = {key: kwargs[self._variable_names[key]] for key in self._variable_names}
        return output

    def rescale(self, val, **required_variables):
        self.update_conditions(**required_variables)
        if self.alpha < 1:
            return self.minimum
        else:
            return super(VariableBeta, self).rescale(val)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)


class Discrete(Prior):

    def __init__(self, minimum, maximum, step_size, name=None,
                 latex_label=None, boundary=None):
        super(Discrete, self).__init__(
            name=name, latex_label=latex_label, boundary=boundary)
        self.minimum = minimum
        self.maximum = maximum
        self.step_size = step_size
        if (maximum - minimum + 1) % step_size != 0:
            raise ValueError(
                'maximum - minimum must be an integer multiple of step size')

    @property
    def n_bins(self):
        return (self.maximum - self.minimum + 1) / self.step_size

    def prob(self, val):
        prob = 1 / self.n_bins
        return prob

    def rescale(self, val):
        val = np.atleast_1d(val)
        val *= self.step_size * self.n_bins
        val += self.minimum
        if isinstance(val, (float, int)) or len(val) == 1:
            val = int(val)
        else:
            val = val.astype(int)
        return val
