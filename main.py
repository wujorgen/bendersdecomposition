from bendersalgo import BendersDecompositionProblem
from equations import system_of_equations

BP = BendersDecompositionProblem(**system_of_equations())
BP.solve()
BP.report(verbose=True)
