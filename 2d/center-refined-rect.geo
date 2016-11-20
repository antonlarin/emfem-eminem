// Square with refined patch in the middle

outer_lc = 1e-1;
inner_lc = 1e-2;

Point(1) = {0, 0, 0, outer_lc};
Point(2) = {1, 0, 0, outer_lc};
Point(3) = {1, 1, 0, outer_lc};
Point(4) = {0, 1, 0, outer_lc};

Point(5) = {0.4, 0.4, 0, inner_lc};
Point(6) = {0.6, 0.4, 0, inner_lc};
Point(7) = {0.6, 0.6, 0, inner_lc};
Point(8) = {0.4, 0.6, 0, inner_lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};

Line(6) = {5, 6};
Line(7) = {6, 7};
Line(8) = {7, 8};
Line(9) = {8, 5};
Line Loop(10) = {6, 7, 8, 9};

Plane Surface(11) = {5, 10};
Plane Surface(12) = {10};

Physical Line("dirichlet-boundary") = {1, 2, 3, 4};
// Physical Line("third-kind-boundary") = {2, 4};
Physical Surface("Whole region") = {11, 12};

