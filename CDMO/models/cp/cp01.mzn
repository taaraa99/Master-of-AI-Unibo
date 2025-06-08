include "globals.mzn";
include "lex_lesseq.mzn";

/*------------------------------------------------------------------
   1. PARAMETERS
------------------------------------------------------------------*/

/*
   couriers: number of couriers (m)
   items:    number of items (n)
   courier_capacity[c] : capacity of courier c
   item_size[i]        : size of item i
   distance_matrix[a,b]: distance from node a to node b.
   There are n+1 nodes total, where nodes 1..n are items and (n+1) is the origin.
*/
int: couriers;
set of int: COURIERS = 1..couriers;
array[COURIERS] of int: courier_capacity;

int: items;
set of int: ITEMS = 1..items;
array[ITEMS] of int: item_size;

set of int: NODES = 1..(items+1);
array[NODES, NODES] of int: distance_matrix;

bool: USE_SB;
bool: USE_IMP;

/*------------------------------------------------------------------
   2. DECISION VARIABLES
------------------------------------------------------------------*/
/* assign[c,i] = true if courier c is assigned to deliver item i. */
array[COURIERS, ITEMS] of var bool: assign;

/* nextStop[c,n] = the next node in courier c's route after node n.
   If nextStop[c,n] = n then node n is not used by courier c (a self-loop);
   otherwise, it points to a different node in 1..(n+1). */
array[COURIERS, NODES] of var NODES: nextStop;

/*------------------------------------------------------------------
   3. CORE CONSTRAINTS
------------------------------------------------------------------*/

/* -- Capacity constraints: total size assigned to a courier must not exceed its capacity. -- */
constraint
  forall(c in COURIERS) (
    sum(i in ITEMS)(bool2int(assign[c,i]) * item_size[i]) <= courier_capacity[c]
  );

/* -- Exactly one courier is assigned to each item. -- */
constraint
  forall(i in ITEMS) (
    sum(c in COURIERS)(bool2int(assign[c,i])) = 1
  );

/* -- Link assign and nextStop:
       - If courier c is not assigned item i then nextStop[c,i] must equal i (self-loop).
       - If courier c is assigned item i then nextStop[c,i] must differ from i.
*/
constraint
  forall(c in COURIERS, i in ITEMS) (
    (assign[c,i] = false) -> (nextStop[c,i] = i)
  );
constraint
  forall(c in COURIERS, i in ITEMS) (
    (assign[c,i] = true) -> (nextStop[c,i] != i)
  );

/* Define a helper Boolean variable used[c] that is true if courier c is assigned at least one item. */
array[COURIERS] of var bool: used;
constraint
  forall(c in COURIERS) (
    used[c] <-> (sum(i in ITEMS)(bool2int(assign[c,i])) > 0)
  );

/* If a courier is used then its route must include the origin.
   That is, nextStop[c, items+1] must differ from (items+1) when used[c] is true;
   otherwise, it must be a self-loop.
*/
constraint
  forall(c in COURIERS) (
    used[c] -> (nextStop[c, items+1] != (items+1))
  );
constraint
  forall(c in COURIERS) (
    not used[c] -> (nextStop[c, items+1] = (items+1))
  );

/* For each courier, nextStop must form a subcircuit (a cycle) over the “active” nodes.
   Nodes that are not used appear as self-loops.
*/
constraint
  forall(c in COURIERS) (
    subcircuit(nextStop[c, ..])
  );

/* ------------------ Optional: Symmetry-Breaking (SB) ------------------ */
constraint
  if USE_SB then
    forall(c1, c2 in COURIERS where c1 < c2) (
      let {
        var int: load_c1 = sum(i in ITEMS)(bool2int(assign[c1,i]) * item_size[i]),
        var int: load_c2 = sum(i in ITEMS)(bool2int(assign[c2,i]) * item_size[i])
      } in
      (load_c1 > 0 /\ load_c2 > 0 /\ load_c1 <= courier_capacity[c2] /\ load_c2 <= courier_capacity[c1])
      -> lex_lesseq(nextStop[c1, ..], nextStop[c2, ..])
    )
  else
    true
  endif;

/* ------------------ Optional: Implied Constraints (IMP) ------------------ */
constraint
  if USE_IMP then
    (
      forall(c in COURIERS, i in ITEMS)(
         item_size[i] > courier_capacity[c] -> assign[c,i] = false
      )
      /\
      (sum(i in ITEMS)(item_size[i]) <= sum(c in COURIERS)(courier_capacity[c]))
    )
  else
    true
  endif;

/*------------------------------------------------------------------
   4. OBJECTIVE: Minimize the maximum route distance
------------------------------------------------------------------*/

/* For each courier, compute routeCost as the sum of distances along its cycle. */
array[COURIERS] of var 0..10000000: routeCost;
constraint
  forall(c in COURIERS) (
    routeCost[c] = sum(n in NODES where nextStop[c,n] != n)(
      distance_matrix[n, nextStop[c,n]]
    )
  );

/* The objective bestObj is the maximum route cost among all couriers. */
var 0..10000000: bestObj = max(c in COURIERS)(routeCost[c]);

/*------------------------------------------------------------------
   5. SEARCH STRATEGY
------------------------------------------------------------------*/
/*
   We use seq_search to first assign items and then determine the routing.
*/
solve :: seq_search([
         int_search([assign[c,i] | c in COURIERS, i in ITEMS], dom_w_deg, indomain_min, complete),
         int_search([nextStop[c,n] | c in COURIERS, n in NODES], dom_w_deg, indomain_min, complete)
      ])
      :: restart_luby(100)
      minimize bestObj;

/*------------------------------------------------------------------
   6. OUTPUT
------------------------------------------------------------------*/
/*
   Instead of printing the "assigned items" (which may be empty because the assignment is encoded
   indirectly via the nextStop routes), we output the nextStop array for each courier.
   This displays the route that each courier follows, similar to the working model.
*/
output [
  "bestObj = " ++ show(bestObj) ++ "\n\n"
]
++
[
  "nextStop for courier " ++ show(c) ++ ": " ++ 
    show([ nextStop[c,n] | n in NODES ]) ++ "\n"
  | c in COURIERS
];
