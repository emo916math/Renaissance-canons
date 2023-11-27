r"""
Canon_flexibility_calculator.sage: a program to generate canons in whole notes
according to simplified Renaissance harmonic rules, and to compute the
flexibility value of a given canon.

EXAMPLES:

- fifthDown = CanonicScheme([(0,0), (1,3)], bass = 1)
  fourthUp = CanonicScheme([(1,3), (0,0)], bass = 1)
  josquin = CanonicScheme([(1,0), (3,3), (0,0), (2,3)], bass = 3)
  josquinDouble = CanonicScheme([(2,0), (6,3), (0,0), (4,3)], bass = 3)
  fib_canon = CanonicScheme([(0,0),(1,-8),(3,0)], bass = 1)
  primi_toni = CanonicScheme([(0,0),(7,-7),(8,-4)])
  sicut_cervus = CanonicScheme([(7,0), (4,4), (0,0)], bass = 2)
  sicut_cervus_p = CanonicScheme([(7,0), (4,4), (0,0)], bass = 2, parallel = True)
  sicut_cervus_r = sicut_cervus.retrograde();
  
  print("Computing flex of Sicut Cervus scheme");
  sicut_cervus.flex();
  sicut_cervus.display(sicut_cervus.random_canon(40))
"""

F7 = GF(7);

class CanonicScheme:
  _tp = [];
  _bass = None;
  _parallel = False;
  
  _mat = None;
  _graph = None;
  _nodes = None;
  _edges = None;
  
  _flex = None;
  _weights = None;
  _Vn_cache = {0:1, 1:7};
  
  """
  Creates a new canonic scheme.
  
  INPUT:
  
  - ``tp`` -- a list of pairs (t,p) representing time and pitch displacements
    for each voice. The t-values should be nonnegative integers; the p-values
    can be integers or elements of the finite field GF(7).
  - ``bass`` -- which voice to choose as bass, (i.e. the other voices cannot be
    a fourth above it), or ``None`` to choose no such voice (as in an
    accompanied canon). Voices are numbered starting from 0 for the first voice.
  - ``parallel`` -- whether to enforce the prohibition on parallel octaves and
    fifths.
    
  """
  def __init__(self, tp, bass = None, parallel = False):
    self._tp = [(ZZ(t), F7(p)) for (t,p) in tp]
    self._bass = bass;
    self._parallel = parallel;
  
  """
  Returns a string representation of this canonic scheme.
  """
  def __repr__(self):
    ret = "CanonicScheme(" + repr(self._tp);
    if self._bass is not None:
      ret += ", bass = " + str(self._bass);
    if self._parallel is not False:
      ret += ", parallel = " + str(self._parallel);
    ret += ")";
    return ret;
  
  """
  Displays the given ``melody``, a list of integers or elements of GF(7),
  as a canon with the scheme given by ``self``.
  """
  def display(self, melody):
    for (t,p) in self._tp:
      print(" "*t, end="");
      for x in melody:
        print(F7(x + p), end="");
      print("");
  
  """
  Checks whether the first and last notes of the given ``melody`` are valid for
  this scheme. A helper method for ``is_valid``, ``all_canons`` and ``random_canon``.
  """
  
  def ends_valid(self, melody, verbose=False):
    tp = self._tp;
    # self.display(melody);
    melody = [F7(x) for x in melody]
    mlen = len(melody);
    for i in [0 .. len(tp) - 2]:
      for j in [i+1 .. len(tp) - 1]:
        diff = tp[j][0] - tp[i][0];
        dissTest = False;
        parallelTest = False;
        if diff == mlen - 1:
          dissTest = True;
          interval = (melody[-1] + tp[i][1]) - (melody[0] + tp[j][1]);
        elif diff == -mlen + 1:
          dissTest = True;
          interval = (melody[0] + tp[i][1]) - (melody[-1] + tp[j][1]);
        elif self._parallel and diff == mlen - 2:
          parallelTest = True;
        elif self._parallel and diff == -mlen + 2:
          parallelTest = True;
        if dissTest:
          if interval == 1:
            if verbose:
              print("Second")
            return False;
          elif interval == -1:
            if verbose:
              print("Seventh")
            return False;
          elif interval == 3 and j == self._bass:
            if verbose:
              print("Fourth")
            return False;
          elif interval == 4 and i == self._bass:
            if verbose:
              print("Fourth")
            return False;
        if parallelTest:
           if len(melody) >= 2 and melody[0] != melody[1]:
             if melody[0] - melody[1] == melody[-2] - melody[-1]:
               interval = (
                 (melody[-2] + tp[i][1]) - (melody[0] + tp[j][1])
                 if diff > 0 else
                 (melody[0] + tp[i][1]) - (melody[-2] + tp[j][1]));
               if interval == 0:
                 if verbose:
                   print("Parallel octaves")
                 return False;
               elif interval == 4:
                 if verbose:
                   print("Parallel fifths")
                 return False;
    # print("Good");  
    return True;
  
  """
  Determines whether the given melody is a valid canon for this scheme.
  If the optional argument ``verbose`` is set to ``True``, then a reason for
  the invalidity is printed if it is not valid.
  """
  def is_valid(self, melody, verbose=False):
    for k in [0..len(melody) - 1]:
      for l in [k+1..len(melody)]:
        if not self.ends_valid(melody[k:l], verbose=verbose):
          return False;
    return True;
  
  """
  Computes the set of all valid canons for this scheme of length ``n`` with
  first note 0 (so the returned set is 7 times smaller than the actual set of
  valid canons). If the optional argument ``double`` is set to ``True'', returns
  the canons of lengths  ``n`` and ``n-1`` as a pair of sets.
  """
  def all_canons(self, n, double=False, verbose=False):
    if n <= 0:
      raise ValueError(n)
    ret = {(F7(),)};
    for nn in [2..n]:
      prev = ret;
      ret = set();
      for mel in prev:
        for add in F7:
          new = mel + (add,);
          if (tuple(new[i] - new[1] for i in [1..len(new) - 1]) in prev
              and self.ends_valid(new)):
            ret.add(new);
      self._Vn_cache.update({nn : 7*len(ret)});
      if verbose:
        print("For length", nn, "there are", 7*len(ret), "canons");
    if double:
      return ret, prev;
    return ret;
  
  """
  Computes the total time displacement of this canonic scheme (``s`` in the
  notation of Theorem 6.1 in the paper).
  """
  def time_disp(self):
    tlist = [t for (t,p) in self._tp];
    if self._parallel:
      time_disp = max(tlist) - min(tlist) + 1;
    else:
      time_disp = max(tlist) - min(tlist);
    return time_disp;
  
  """
  Computes the matrix of this scheme (A_S in the paper)
  """
  def matrix(self):
    if self._mat is not None:
      return self._mat;
    
    M = self.graph().adjacency_matrix()
    
    self._mat = M;
    return M;
  
  """
  Computes the graph of this scheme (G_S in the paper)
  """
  def graph(self, verbose=False):
    if self._graph is not None:
      return self._graph;
    
    time_disp = self.time_disp();
    edges, nodes = self.all_canons(time_disp + 1, double=True, verbose=verbose);
    self._nodes = list(nodes)
    self._edges = list(edges)
    self._graph = DiGraph(
      [nodes, [(edge[:-1], tuple(x - edge[1] for x in edge[1:]), edge)
      for edge in edges]],
        format='vertices_and_edges', loops=True, multiedges=True,
        immutable=True)
    return self._graph
  """
  Computes the number of valid canons of length n for this scheme (V_n(S) in the
  paper). For small n, this is done by enumerating all valid canons. For n at
  least the time order s, we do it by computing powers of the adjacency matrix
  as in Section 6 of the paper.
  """
  def Vn(self, n, float_out=False):
    if n not in ZZ or n < 0:
      raise ValueError(n);
    cached = self._Vn_cache.get(n);
    if cached is not None:
      return cached;
    s = self.time_disp();
    if n < s:
      return 7*len(self.all_canons(n));
    if float_out:
      M = self.matrix().change_ring(RR);
      all_ones = vector([1.0]*M.nrows());
    else:
      M = self.matrix();
      all_ones = vector([1]*M.nrows());
    if n-s > M.nrows():
      ret = 7*all_ones * M^(n-s) * all_ones;
      self._Vn_cache.update({n : ret});
      return ret;
    else:
      v = all_ones;
      for it in range(s,n):
        v = v*M;
        if self._Vn_cache.get(it) is not None:
          self._Vn_cache.update({it : 7*sum(v)})
      return 7*sum(v);
    
  """
  Computes the sizes of the strongly connected components of this scheme
  (G_i in the paper)
  """
  def component_sizes(self):
    g = self.graph();
    return [len(sg) for sg in g.strongly_connected_components()]
  
  """
  Computes the flexibility value of this scheme (lambda(S) in the paper).
  
  OUTPUT:
  - a real number, the flexibility value lambda
  - a dictionary of mappings (melody -> weight), giving the entries of a
  dominant eigenvector (v_i in Section A.2 of the paper)
  """
  def flex(self, verbose = True):
    if self._flex is not None:
      return self._flex, self._weights;
    
    max_flex = 0; weights = {};
    for sg in (self.graph(verbose = verbose)
        .strongly_connected_components_subgraphs()):
      flx, wts = CanonicScheme.get_flex(sg, verbose = verbose)
      max_flex = max(max_flex, flx);
      weights.update(wts)
      
    self._flex = max_flex;
    self._weights = weights;
    
    if verbose:
      print("Flexibility is", max_flex);
    return max_flex, self._weights;
    
  """
  Computes the dominant eigenvalue and eigenvector of a given graph. A helper
  method for flex().
  """
  def get_flex(graph, verbose = True):
    nodes = graph.vertices(sort = False);
    if verbose and len(nodes) > 1:
      print("Component with", len(graph), "vertices:", end=" ", flush=True)

    M = graph.adjacency_matrix();
    
    # Use power method to approximate the dominant eigenvector.
    v = vector([RDF(1)]*M.nrows());
    found = False;
    for i in [1..3000]: # ?
      v2 = M*v;
      if i % 10 == 0:
        eigs = [v2[i] / v[i] for i in range(len(v)) if v[i] != 0];
        mxeig = max(eigs)
        mneig = min(eigs);
        if mxeig - mneig < 1.0e-13:
          found = True;
          break;
      if sum(v2) == 0:
        return 0.0, {node : 0 for node in nodes};
      v = v2 / sum(v2); # to avoid overflow
      
    if not(found):
      print("warning: error", mxeig - mneig)
    weights = {nodes[i] : v[i] for i in range(len(v))};
    flex = mxeig;
    if verbose:
      if len(nodes) == 1:
        print("Component with 1 vertex:", end=" ", flush=True)
      print("eigenvalue is", flex);
    return flex, weights;
  
  """
  Computes a random canon for this scheme of a given length ``n``. The
  possibilities are weighted according to the dominant eigenvector; if the
  dominant eigenvalue is simple (as is usually the case), this is equivalent to
  taking a block of length ``n`` from a canon chosen uniformly from all valid
  canons of length ``N`` and letting ``N`` tend to infinity.
  """
  def random_canon(self, n, start=(0,), display = False):
    start = tuple(start)
    self.graph();
    nodes = self._nodes;
    edges = self._edges;
    flex, weights = self.flex()
    
    s = self.time_disp();
    if len(start) < s:
      start0 = tuple(x - start[0] for x in start)
      snodes = [node for node in nodes if node[0:len(start)] == start0];
      if len(snodes) == 0:
        raise ValueError("No canons that start with " + str(start)) 
      vec = vector(weights.get(n, 0) for n in snodes)
      vec /= sum(vec);
      rand = random()
      total = 0; idx = -1;
      while total < rand:
        idx += 1;
        total += vec[idx];
      ret = tuple(x + start[0] for x in snodes[idx]);
      node = snodes[idx];
    else:
      ret = start;
      node = tuple(x - start[-s] for x in start[-s:])
    
    while len(ret) < n:
      edg = self.graph().outgoing_edges(node);
      snodes = [out for (in_, out, edge) in edg];
      if len(snodes) == 0:
        raise ValueError("No canons that start with " + str(ret)) 
      vec = vector(weights.get(n, 0) for n in snodes)
      vec /= sum(vec);
      rand = random()
      total = 0; idx = -1;
      while total < rand:
        idx += 1;
        total += vec[idx];
      ret += (snodes[idx][-1] - snodes[idx][0] + ret[-s + 1],);
      node = snodes[idx];
      
    if display:
      self.display(ret);
    return ret;
  
  """
  Computes the retrograde of this scheme, that is, a scheme with the same pitch
  displacements but with the time displacements reversed.
  """
  def retrograde(self):
    tlist = [t for (t,p) in self._tp];
    max_t = max(tlist)
    return CanonicScheme([(max_t, p) for (t,p) in self._tp],
      bass = self._bass, parallel = self._parallel);
        
