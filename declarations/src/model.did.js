export const idlFactory = ({ IDL }) => {
  const ObservationVector = IDL.Record({
    'i' : IDL.Float64,
    'j' : IDL.Float64,
    'k' : IDL.Float64,
    'lat' : IDL.Float64,
    'lon' : IDL.Float64,
  });
  return IDL.Service({
    'a_add_authorized_user' : IDL.Func([IDL.Principal], [], []),
    'a_initialize_authorized_user' : IDL.Func([IDL.Principal], [], []),
    'a_list_authorized_users' : IDL.Func(
        [],
        [IDL.Vec(IDL.Principal)],
        ['query'],
      ),
    'a_remove_authorized_user' : IDL.Func([IDL.Principal], [], []),
    'm_fit_obs' : IDL.Func([IDL.Vec(ObservationVector)], [], []),
    'm_fit_pred' : IDL.Func([], [], []),
    'm_predict' : IDL.Func([IDL.Bool], [], []),
    'm_scores' : IDL.Func([], [IDL.Vec(IDL.Nat16)], []),
  });
};
export const init = ({ IDL }) => { return []; };
