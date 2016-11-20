// Sample problem with axial symmetry

coarse_lc = 2e-2;
fine_lc = 1e-2;

Point(1) = {0, 0.2, 0, coarse_lc};
Point(2) = {0.5, 0.2, 0, coarse_lc};
Point(3) = {0.5, 0.4, 0, fine_lc};
Point(4) = {1, 0.4, 0, coarse_lc};
Point(5) = {1, 0.6, 0, coarse_lc};
Point(6) = {0, 0.6, 0, coarse_lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Line Loop(7) = {1, 2, 3, 4, 5, 6};
Plane Surface(8) = {7};

Physical Line("dirichlet-boundary") = {1, 2, 3, 5};
Physical Line("third-kind-boundary") = {4, 6};
Physical Surface("whole-domain") = {8};

