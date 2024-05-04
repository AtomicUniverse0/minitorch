from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter] # parameters 可以当做一个简单的容器来看待……
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        "Return the direct child modules of this module."
        m: Dict[str, Module] = self.__dict__["_modules"] # 不懂这是在干什么…… 为什么不直接self._modules呢？
        return list(m.values())

    def train(self) -> None:
        "Set the mode of this module and all descendent modules to `train`."
        # TODO: Implement for Task 0.4.
        self.training = True
        for key in self._modules:
            self._modules[key].train()


    def eval(self) -> None:
        "Set the mode of this module and all descendent modules to `eval`."
        # TODO: Implement for Task 0.4.
        self.training = False
        for key in self._modules:
            self._modules[key].eval()

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """
        Collect all the parameters of this module and its descendents.


        Returns:
            The name and `Parameter` of each ancestor parameter.
        """
        # TODO: Implement for Task 0.4.
        res = []

        # 自己的参数
        for str, par in self._parameters.items():
            res.append((str, par))
        
        # 孩子们的参数
        for module_name, module in self._modules.items():

            temp = module.named_parameters()
            for tuple in temp:
                res.append((module_name + "." + tuple[0], tuple[1])) # 每次都复制一遍，开销会不会太大……
            
        
        return res
        

    def parameters(self) -> Sequence[Parameter]:
        "Enumerate over all the parameters of this module and its descendents."
        # TODO: Implement for Task 0.4.
        res = []

        for _, par in self._parameters.items():
            res.append(par)

        for _, des in self._modules.items():
            res += des.parameters()

        return res

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
            Newly created parameter.
        """
        val = Parameter(v, k) # key 作为了 Parameter的name
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None: # 重载=
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"): # hasattr： 判断对象是否包含对应的属性
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
