import numpy as np
from bilby.core.prior import Prior, Uniform, Beta
from bilby.core.utils import infer_args_from_method, infer_parameters_from_function


class ConditionalBeta(Beta):
    def __init__(self, condition_func, name=None, latex_label=None, unit=None,
                 boundary=None, **reference_params):
        """

        Parameters
        ----------
        condition_func: func
            Functional form of the condition for this prior. The first function argument
            has to be a dictionary for the `reference_params` (see below). The following
            arguments are the required variables that are required before we can draw this
            prior.
            It needs to return a dictionary with the modified values for the
            `reference_params` that are being used in the next draw.
            For example if we have a Uniform prior for `x` depending on a different variable `y`
            `p(x|y)` with the boundaries linearly depending on y, then this
            could have the following form:

            ```
            def condition_func(reference_params, y):
                return dict(minimum=reference_params['minimum'] + y, maximum=reference_params['maximum'] + y)
            ```
        name: str, optional
           See superclass
        latex_label: str, optional
            See superclass
        unit: str, optional
            See superclass
        boundary: str, optional
            See superclass
        reference_params:
            Initial values for attributes such as `minimum`, `maximum`.
            This differs on the `prior_class`, for example for the Gaussian
            prior this is `mu` and `sigma`.
        """
        if 'boundary' in infer_args_from_method(super(ConditionalBeta, self).__init__):
            super(ConditionalBeta, self).__init__(
                name=name, latex_label=latex_label, unit=unit,
                boundary=boundary, **reference_params)
        else:
            super(ConditionalBeta, self).__init__(
                name=name, latex_label=latex_label, unit=unit,
                **reference_params)

        self._required_variables = None
        self.condition_func = condition_func
        self._reference_params = reference_params

        # These lines break the pickling for multiprocessing
        # self.__class__.__name__ = 'Conditional{}'.format(Beta.__name__)
        # self.__class__.__qualname__ = 'Conditional{}'.format(Beta.__qualname__)

    def sample(self, size=None, **required_variables):
        """Draw a sample from the prior

        Parameters
        ----------
        size: int or tuple of ints, optional
            See superclass
        required_variables:
            Any required variables that this prior depends on

        Returns
        -------
        float: See superclass

        """
        self.least_recently_sampled = self.rescale(np.random.uniform(0, 1, size), **required_variables)
        return self.least_recently_sampled

    def rescale(self, val, **required_variables):
        """
        'Rescale' a sample from the unit line element to the prior.

        Parameters
        ----------
        val: Union[float, int, array_like]
            See superclass
        required_variables:
            Any required variables that this prior depends on


        """
        self.update_conditions(**required_variables)
        return super(ConditionalBeta, self).rescale(val)

    def prob(self, val, **required_variables):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]
            See superclass
        required_variables:
            Any required variables that this prior depends on


        Returns
        -------
        float: Prior probability of val
        """
        self.update_conditions(**required_variables)
        return super(ConditionalBeta, self).prob(val)

    def ln_prob(self, val, **required_variables):
        self.update_conditions(**required_variables)
        return super(ConditionalBeta, self).ln_prob(val)

    def update_conditions(self, **required_variables):
        """
        This method updates the conditional parameters (depending on the parent class
        this could be e.g. `minimum`, `maximum`, `mu`, `sigma`, etc.) of this prior
        class depending on the required variables it depends on.

        If no variables are given, the most recently used conditional parameters are kept

        Parameters
        ----------
        required_variables:
            Any required variables that this prior depends on. If none are given,
            self.reference_params will be used.

        """
        if sorted(list(required_variables)) == sorted(self.required_variables):
            parameters = self.condition_func(self.reference_params, **required_variables)
            for key, value in parameters.items():
                setattr(self, key, value)
        elif len(required_variables) == 0:
            return
        else:
            raise IllegalRequiredVariablesException("Expected kwargs for {}. Got kwargs for {} instead."
                                                    .format(self.required_variables,
                                                            list(required_variables.keys())))

    @property
    def reference_params(self):
        """
        Initial values for attributes such as `minimum`, `maximum`.
        This depends on the `prior_class`, for example for the Gaussian
        prior this is `mu` and `sigma`. This is read-only.
        """
        return self._reference_params

    @property
    def condition_func(self):
        return self._condition_func

    @condition_func.setter
    def condition_func(self, condition_func):
        if condition_func is None:
            self._condition_func = lambda reference_params: reference_params
        else:
            self._condition_func = condition_func
        self._required_variables = infer_parameters_from_function(self.condition_func)

    @property
    def required_variables(self):
        """ The required variables to pass into the condition function. """
        return self._required_variables

    def get_instantiation_dict(self):
        instantiation_dict = super(ConditionalBeta, self).get_instantiation_dict()
        for key, value in self.reference_params.items():
            instantiation_dict[key] = value
        return instantiation_dict

    def reset_to_reference_parameters(self):
        """
        Reset the object attributes to match the original reference parameters
        """
        for key, value in self.reference_params.items():
            setattr(self, key, value)


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
        return dict(minimum=kwargs[self.reference_name] + 1 / self.duration)

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
