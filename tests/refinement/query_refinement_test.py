from services.query_refinement_service import (
    refine_query
)

query = input(
    "Enter Query: "
)

print(
    "\nRefined Query:"
)

print(
    refine_query(query)
)