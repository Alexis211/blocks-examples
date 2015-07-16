import logging

from blocks.algorithms import GradientDescent

logger = logging.getLogger(__name__)


class CustomizedGradientDescent(GradientDescent):
    """
    This class allows the gradient descent to use a custom step rule
    for different variables. To use a custom step rule on a given parameter,
    simply set its `custom_step_rule` tag to a step rule class. Setting a
    value of None will disable gradient descent and optimization on that variable.
    """
    def __init__(self, parameters, **kwargs):

        parameters = filter(lambda p: not hasattr(p.tag, 'custom_step_rule')
                                      or p.tag.custom_step_rule is not None,
                            parameters)

        super(CustomizedGradientDescent, self).__init__(parameters=parameters,
                                                    **kwargs)

        self.steps = {}
        self.step_rule_updates = []

        for param in self.parameters:
            if hasattr(param.tag, 'custom_step_rule'):
                step_rule = param.tag.custom_step_rule
            else:
                step_rule = self.step_rule
            
            assert step_rule is not None
            logger.info("Calculating update for {} using step rule {}".format(
                        param.name, repr(type(step_rule))))

            steps, updates = step_rule.compute_steps({param: self.gradients[param]})
            self.steps = dict(self.steps.items() + steps.items())
            self.step_rule_updates += updates

